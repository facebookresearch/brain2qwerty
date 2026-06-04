# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import lightning.pytorch as pl
import neuralset as ns
import pydantic
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from neuraltrain.losses import LossConfig
from neuraltrain.metrics import MetricConfig
from neuraltrain.models import ModelConfig
from neuraltrain.optimizers import LightningOptimizerConfig
from neuraltrain.utils import BaseExperiment, WandbInfra
from torch import nn
from torch.utils.data import DataLoader

from .callbacks import LogSentencePredictions
from .metrics import SentenceCER
from .pl_module import BrainModule
from .splitter import split_events
from .utils import ShuffledSegmentDataset, preprocessing


class Data(pydantic.BaseModel):
    """Configuration and creation of DataLoaders from MEG/EEG data."""

    model_config = pydantic.ConfigDict(extra="forbid")

    study: ns.data.StudyLoader
    neuro: ns.features.FeatureConfig
    feature: ns.features.FeatureConfig
    num_classes: int = 29

    start: float = -0.2
    duration: float = 0.5
    batch_size: int = 128
    val_batch_size: int = 2048
    test_batch_size: int = 2048
    num_workers: int = 0

    splitting_seed: int | None = None
    splitting_ratios: tuple = (0.8, 0.1, 0.1)

    def build(self) -> dict[str, DataLoader]:
        neuro_type = self.neuro.event_types

        events = self.study.build()
        events = preprocessing(events)
        events = split_events(events, self.splitting_ratios, self.splitting_seed)

        self.neuro.prepare(events)
        self.feature.prepare(events)

        subject_id = ns.features.LabelEncoder(
            event_types=neuro_type, event_field="subject"
        )
        subject_id.__class__.event_types = getattr(ns.events, neuro_type)
        subject_id.prepare(events)

        features = {
            "neuro": self.neuro,
            "feature": self.feature,
            "subject_id": subject_id,
        }

        if neuro_type in ["Meg", "Eeg"]:
            channel_positions = ns.features.ChannelPositions(neuro=self.neuro)
            channel_positions.prepare(events)
            features["channel_positions"] = channel_positions

        loaders = {}
        batch_sizes = {
            "train": self.batch_size,
            "val": self.val_batch_size,
            "test": self.test_batch_size,
        }

        for split, batch_size in batch_sizes.items():
            split_mask = (events.split == split) & (events.type == "Button")
            segments = ns.segments.list_segments(
                events,
                split_mask,
                start=self.start,
                duration=self.duration,
            )
            dataset = ShuffledSegmentDataset(
                features=features,
                segments=segments,
                remove_incomplete_segments=True,
            )
            loaders[split] = DataLoader(
                dataset,
                collate_fn=dataset.collate_fn,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        return loaders


class Experiment(BaseExperiment):
    """Main experiment: data loading, model training, and evaluation."""

    data: Data

    seed: int = 33
    brain_model_config: ModelConfig
    transformer_config: ModelConfig
    use_transformer: bool = True
    loss: LossConfig
    metrics: list[MetricConfig]
    optimizer: LightningOptimizerConfig
    save_checkpoints: bool = True

    n_epochs: int = 100
    patience: int = 80
    lr: float = 1e-4
    grad_max_norm: float = 5.0
    weight_decay: float = 1e-4

    strategy: str = "auto"
    accelerator: str = "gpu"
    log_every_n_steps: int | None = 5
    limit_train_batches: int | None = None
    fast_dev_run: bool = False

    infra: WandbInfra = WandbInfra(version="1")

    _trainer: pl.Trainer | None = None
    _brain_module: BrainModule | None = None
    _logger: WandbLogger | None = None

    def model_post_init(self, __context: tp.Any) -> None:
        assert self.infra.folder is not None, "infra.folder must be specified."

    def _init_module(
        self, model: nn.Module, transformer: nn.Module | None
    ) -> BrainModule:
        checkpoint_path = Path(self.infra.folder) / "last.ckpt"
        if checkpoint_path.exists():
            init_fn = BrainModule.load_from_checkpoint
        else:
            init_fn = BrainModule
            checkpoint_path = None

        return init_fn(
            model=model,
            transformer=transformer,
            loss=self.loss.build(),
            metrics={
                **{m.log_name: m.build() for m in self.metrics},
                "CER": SentenceCER(),
            },
            lr=self.lr,
            max_epochs=self.n_epochs,
            grad_max_norm=self.grad_max_norm,
            weight_decay=self.weight_decay,
            optimizer=self.optimizer,
            checkpoint_path=checkpoint_path,
        )

    def _setup_trainer(self) -> pl.Trainer:
        callbacks = [
            LogSentencePredictions(),
            EarlyStopping(monitor="val_CER", mode="min", patience=self.patience),
        ]
        if self.save_checkpoints:
            callbacks.append(
                ModelCheckpoint(
                    save_last=True,
                    save_top_k=1,
                    dirpath=self.infra.folder,
                    filename="best",
                    monitor="val_CER",
                    mode="min",
                    save_on_train_epoch_end=True,
                )
            )

        return pl.Trainer(
            strategy=self.strategy,
            devices=self.infra.gpus_per_node,
            accelerator=self.accelerator,
            max_epochs=self.n_epochs,
            limit_train_batches=self.limit_train_batches,
            enable_progress_bar=True,
            log_every_n_steps=self.log_every_n_steps,
            fast_dev_run=self.fast_dev_run,
            callbacks=callbacks,
            logger=self._logger,
        )

    def fit(self, train_loader: DataLoader, valid_loader: DataLoader) -> None:
        batch = next(iter(train_loader))
        n_in_channels = batch.data["neuro"].shape[1]
        n_hidden = self.brain_model_config.hidden

        brain_model = self.brain_model_config.build(
            n_in_channels=n_in_channels, n_outputs=n_hidden
        )
        transformer = self.transformer_config.build(dim=n_hidden)

        self._brain_module = self._init_module(brain_model, transformer)
        self._trainer = self._setup_trainer()
        self._trainer.fit(
            model=self._brain_module,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
            ckpt_path=self._brain_module.checkpoint_path,
        )

    def test(self, test_loader: DataLoader) -> None:
        self._trainer.test(self._brain_module, dataloaders=test_loader)

    @infra.apply
    def run(self):
        self._logger = (
            self.infra.wandb_config.build(
                save_dir=self.infra.folder,
                xp_config=self.model_dump(),
            )
            if self.infra.wandb_config
            else None
        )
        pl.seed_everything(self.seed, workers=True)
        loaders = self.data.build()
        self.fit(loaders["train"], loaders["val"])
        self.test(loaders["test"])
