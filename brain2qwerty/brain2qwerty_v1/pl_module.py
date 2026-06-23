# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from torch import nn
from torchmetrics import Metric

from neuralset.dataloader import Batch

from .utils import NUM_CLASSES


class BrainModule(pl.LightningModule):
    """Convolutional encoder + sentence-level transformer for keystroke decoding.

    The encoder produces one embedding per keystroke window; embeddings are
    grouped by sentence, refined by the transformer, and projected to characters.
    """

    def __init__(
        self,
        model: nn.Module,
        transformer: nn.Module,
        loss: nn.Module,
        metrics: dict[str, Metric],
        optimizer,
        x_name: str = "neuro",
        y_name: str = "feature",
        checkpoint_path: Path | None = None,
    ):
        super().__init__()
        self.model = model
        self.transformer = transformer
        self.linear = nn.Linear(model.out_channels, NUM_CLASSES)
        self.x_name, self.y_name = x_name, y_name
        self.loss = loss
        self.optimizer = optimizer
        self.checkpoint_path = checkpoint_path
        self.metrics = nn.ModuleDict(
            {f"{split}_{k}": v for k, v in metrics.items() for split in ["val", "test"]}
        )
        self.save_hyperparameters(ignore=["model", "transformer", "loss"])

    def forward(self, batch: Batch) -> torch.Tensor:
        return self.model(
            batch.data[self.x_name],
            batch.data["subject_id"],
            batch.data["channel_positions"],
        )

    def _transformer_forward(self, batch: Batch, y_pred: torch.Tensor) -> torch.Tensor:
        uids = np.array([seg.trigger.extra["sentence_UID"] for seg in batch.segments])
        unique_uids, first_idx = np.unique(uids, return_index=True)
        unique_uids = unique_uids[np.argsort(first_idx)]

        grouped = [
            torch.stack([y_pred[i] for i, s in enumerate(uids) if s == uid])
            for uid in unique_uids
        ]
        max_len = max(len(g) for g in grouped)
        x = torch.zeros(len(grouped), max_len, y_pred.shape[1], device=y_pred.device)
        mask = torch.zeros(len(grouped), max_len, device=y_pred.device)
        for i, g in enumerate(grouped):
            x[i, : len(g)] = g
            mask[i, : len(g)] = 1

        out = self.transformer(x, mask=mask.bool())
        flat = [out[i][: len(g)] for i, g in enumerate(grouped)]
        return self.linear(torch.cat(flat))

    def _run_step(self, batch: Batch, step_name: str):
        y_true = batch.data[self.y_name].squeeze(1)
        y_pred = self._transformer_forward(batch, self.forward(batch))
        loss = self.loss(y_pred, y_true)

        self.log(
            f"{step_name}_loss",
            loss,
            on_step=(step_name == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=y_pred.shape[0],
        )
        for name, metric in self.metrics.items():
            if name.startswith(step_name):
                metric.update(y_pred, y_true)
                self.log(
                    name,
                    metric,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=y_pred.shape[0],
                )
        return loss, y_pred, y_true

    def training_step(self, batch: Batch, batch_idx):
        return self._run_step(batch, "train")[0]

    def validation_step(self, batch: Batch, batch_idx):
        _, y_pred, y_true = self._run_step(batch, "val")
        return y_pred, y_true

    def test_step(self, batch: Batch, batch_idx):
        _, y_pred, y_true = self._run_step(batch, "test")
        return y_pred, y_true

    def configure_optimizers(self):
        return self.optimizer.build(
            self.parameters(), total_steps=self.trainer.estimated_stepping_batches
        )
