# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from neuralset.dataloader import SegmentData
from torch import nn
from torchmetrics import Metric

from .utils import NUM_CLASSES


class BrainModule(pl.LightningModule):
    """PyTorch Lightning module for Brain2Qwerty training.

    Wraps a convolutional encoder and an optional transformer that refines
    keystroke predictions at the sentence level.
    """

    def __init__(
        self,
        model: nn.Module,
        transformer: nn.Module | None,
        loss: nn.Module,
        metrics: dict[str, Metric],
        x_name: str = "neuro",
        y_name: str = "feature",
        lr: float = 1e-4,
        max_epochs: int = 100,
        grad_max_norm: float = 1.0,
        weight_decay: float = 1e-4,
        checkpoint_path: Path | None = None,
        optimizer=None,
    ):
        super().__init__()
        self.model = model
        self.transformer = transformer
        self.linear = nn.Linear(model.out_channels, NUM_CLASSES)
        self.x_name, self.y_name = x_name, y_name
        self.checkpoint_path = checkpoint_path

        self.lr = lr
        self.max_epochs = max_epochs
        self.grad_max_norm = grad_max_norm
        self.weight_decay = weight_decay
        self.optimizer = optimizer

        self.loss = loss
        self.metrics = nn.ModuleDict(
            {f"{split}_{k}": v for k, v in metrics.items() for split in ["val", "test"]}
        )

        self.save_hyperparameters(ignore=["model", "loss"])

    def forward(self, batch: SegmentData) -> torch.Tensor:
        x = batch.data["neuro"]
        subject_ids = batch.data.get("subject_id")
        channel_positions = batch.data.get("channel_positions")
        return self.model(x, subject_ids, channel_positions)

    def _transformer_forward(
        self, batch: SegmentData, y_pred: torch.Tensor
    ) -> torch.Tensor:
        sentence_uids = np.array(
            [
                f"{seg._trigger['trial_id']}_{seg._trigger['timeline']}"
                for seg in batch.segments
            ]
        )
        unique_uids, sentence_idx = np.unique(sentence_uids, return_index=True)
        unique_uids = unique_uids[np.argsort(sentence_idx)]

        grouped = []
        for uid in unique_uids:
            indices = [i for i, s in enumerate(sentence_uids) if s == uid]
            grouped.append(torch.stack([y_pred[i] for i in indices]))

        max_len = max(len(g) for g in grouped)
        transformer_input = torch.zeros(
            len(grouped), max_len, y_pred.shape[1], device=y_pred.device
        )
        mask = torch.zeros(len(grouped), max_len, device=y_pred.device)
        for i, g in enumerate(grouped):
            transformer_input[i, : len(g)] = g
            mask[i, : len(g)] = 1

        transformer_output = self.transformer(transformer_input, mask=mask.bool())

        out = []
        for i, g in enumerate(grouped):
            out.extend(transformer_output[i][: len(g)])
        return self.linear(torch.stack(out))

    def _run_step(self, batch: SegmentData, batch_idx, step_name):
        y_true = batch.data[self.y_name].squeeze(1)
        y_pred = self.forward(batch)

        if self.transformer is not None:
            y_pred = self._transformer_forward(batch, y_pred)
        loss = self.loss(y_pred, y_true)

        self.log(
            f"{step_name}_loss",
            loss,
            on_step=(step_name == "train"),
            on_epoch=True,
            logger=True,
            prog_bar=True,
            batch_size=y_pred.shape[0],
        )

        for metric_name, metric in self.metrics.items():
            if metric_name.startswith(step_name):
                metric.update(y_pred, y_true)
                self.log(
                    metric_name,
                    metric,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    batch_size=y_pred.shape[0],
                )

        return loss, y_pred, y_true

    def training_step(self, batch: SegmentData, batch_idx):
        loss, _, _ = self._run_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch: SegmentData, batch_idx):
        _, y_pred, y_true = self._run_step(batch, batch_idx, "val")
        return y_pred, y_true

    def test_step(self, batch: SegmentData, batch_idx):
        _, y_pred, y_true = self._run_step(batch, batch_idx, "test")
        return y_pred, y_true

    def configure_optimizers(self):
        if hasattr(self.optimizer, "scheduler"):
            self.optimizer.scheduler.kwargs["total_steps"] = (
                self.trainer.estimated_stepping_batches
            )
        return self.optimizer.build(self.parameters())
