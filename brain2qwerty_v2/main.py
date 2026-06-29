# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from pathlib import Path

import lightning.pytorch as pl
import pydantic
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from torchmetrics.text import CharErrorRate, WordErrorRate
from transformers import AutoModelForCausalLM, AutoTokenizer

import neuralset as ns
from neuralset.events.study import EventsTransform
from neuraltrain.models.base import BaseModelConfig
from neuraltrain.utils import WandbLoggerConfig

from . import models as _models  # noqa: F401  (registers the ConvConformer encoder)
from . import transforms as _transforms  # noqa: F401  (registers EventsTransforms)
from .callbacks import PredictionCSVCallback
from .config.xp_config import LLM, WORD_EXTRACTOR
from .data import SentenceDataset
from .metrics import SemanticErrorRate
from .pl_module import NeuroLLMModule
from .utils import ChannelPositions2D, accelerator, build_events, prepare_word_embeddings

log = logging.getLogger(__name__)


class Data(pydantic.BaseModel):
    """Sentence-level dataloaders for Brain2Qwerty V2.

    Runs the study, applies the preprocessing/split/word transforms, extends each
    sentence window by a small random tail, and builds one padded dataloader per
    split (train-time MEG onset jitter is applied inside the dataset).
    """

    model_config = pydantic.ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    study: ns.events.Study
    transforms: list[EventsTransform] = pydantic.Field(default_factory=list)
    neuro: ns.extractors.BaseExtractor
    extractor: ns.extractors.BaseExtractor

    start: float = -0.4
    duration: float | None = None
    jitter: bool = True
    num_classes: int = 29
    tail_min: float = 0.4  # extend each sentence window by a random tail (seconds)
    tail_max: float = 0.5

    batch_size: int = 32
    val_batch_size: int = 128
    test_batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False

    def build(self) -> dict[str, DataLoader]:
        events = build_events(self.study, self.transforms, (self.tail_min, self.tail_max))
        self.neuro.prepare(events)
        self.extractor.prepare(events)

        subject_encoder = ns.extractors.LabelEncoder(
            event_types="Meg", event_field="subject"
        )
        subject_encoder.prepare(events)
        chan_pos = ChannelPositions2D(neuro=self.neuro)
        chan_pos.prepare(events)

        extractors = {
            "neuros": self.neuro,
            "phonemes": self.extractor,
            "days": subject_encoder,
            "chan_pos": chan_pos,
        }
        batch_sizes = {
            "train": self.batch_size,
            "val": self.val_batch_size,
            "test": self.test_batch_size,
        }
        loaders: dict[str, DataLoader] = {}
        for split, batch_size in batch_sizes.items():
            mask = (events.split == split) & (events.type == "Sentence")
            segments = ns.segments.list_segments(
                events, mask, start=self.start, duration=self.duration
            )
            if not segments:
                continue
            dataset = SentenceDataset(
                extractors,
                segments,
                jitter=(self.jitter and split == "train"),
                remove_incomplete_segments=True,
            )
            loaders[split] = DataLoader(
                dataset,
                collate_fn=dataset.collate_fn,
                batch_size=batch_size,
                shuffle=(split == "train"),
                drop_last=(split == "train"),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
            )
        return loaders


class Experiment(pydantic.BaseModel):
    """Train and evaluate the Brain2Qwerty V2 end-to-end pipeline."""

    model_config = pydantic.ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    data: Data
    brain_model_config: BaseModelConfig
    num_classes: int = 29

    seed: int = 123
    max_epochs: int = 275
    precision: str = "16-mixed"
    gradient_clip_val: float | None = 1.0
    accumulate_gradient_batches: int = 2
    devices: int = 8
    output_dir: str = "."

    # loss weighting + staged schedule
    alpha: float = 0.1
    beta: float = 0.01
    loss_alpha: float = 0.7
    ctc_start_epoch: int = 0
    contrastive_start_epoch: int = 150
    llm_start_epoch: int = 225
    encoder_lr: float | None = None

    # contrastive / segmenter
    word_pool_n_layers: int = 2
    seg_include_blanks: bool = True
    word_extractor_config: dict = pydantic.Field(
        default_factory=lambda: dict(WORD_EXTRACTOR)
    )

    # LLM + LoRA
    llm_name: str = LLM
    lora_rank: int = 2
    lora_alpha_value: int = 4
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = pydantic.Field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # generation
    max_new_tokens: int = 60
    num_beams: int = 16
    val_num_beams: int = 1
    length_penalty: float = 0.2
    label_smoothing: float = 0.02
    meg_dropout_rate: float = 0.1
    ctc_dropout_rate: float = 0.1

    optimizer_config: dict = pydantic.Field(
        default_factory=lambda: {"lr": 8e-4, "weight_decay": 1e-3}
    )
    scheduler_config: dict = pydantic.Field(
        default_factory=lambda: {"name": "OneCycleLR", "pct_start": 0.3}
    )
    preprocess_config: dict = pydantic.Field(default_factory=dict)

    save_checkpoints: bool = True
    eval_only: bool = False
    ckpt_path: str | None = None
    resume_ckpt: str | None = None  # resume training (trainer state) from this ckpt
    wandb_config: WandbLoggerConfig | None = None

    _trainer: pl.Trainer | None = None
    _module: NeuroLLMModule | None = None

    def model_post_init(self, log__: tp.Any) -> None:
        pl.seed_everything(self.seed, workers=True)
        torch.set_float32_matmul_precision("medium")

    def _build_module(self, loaders: dict) -> NeuroLLMModule:
        word_embed_lookup = prepare_word_embeddings(self.data, self.word_extractor_config)
        n_in_channels = loaders["train"].dataset[0].data["neuros"].shape[1]
        network = self.brain_model_config.build(
            n_in_channels=n_in_channels, n_outputs=self.num_classes
        )
        word_pool_dim = getattr(self.brain_model_config, "dim", 1024)

        llm = AutoModelForCausalLM.from_pretrained(
            self.llm_name, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        llm = get_peft_model(
            llm,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.lora_rank,
                lora_alpha=self.lora_alpha_value,
                lora_dropout=self.lora_dropout,
                target_modules=list(self.lora_target_modules),
            ),
        )
        llm.print_trainable_parameters()
        tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        llm_hidden = llm.get_base_model().config.hidden_size
        adapter: nn.Module = (
            nn.Linear(word_pool_dim, llm_hidden)
            if word_pool_dim != llm_hidden
            else nn.Identity()
        )
        llm_metrics = {
            "CER": CharErrorRate(),
            "WER": WordErrorRate(),
            "SemER": SemanticErrorRate(),
        }

        module = NeuroLLMModule(
            network=network,
            llm=llm,
            tokenizer=tokenizer,
            word_proj_adapter=adapter,
            word_embed_lookup=word_embed_lookup,
            word_pool_dim=word_pool_dim,
            word_pool_n_layers=self.word_pool_n_layers,
            seg_include_blanks=self.seg_include_blanks,
            alpha=self.alpha,
            beta=self.beta,
            loss_alpha=self.loss_alpha,
            ctc_start_epoch=self.ctc_start_epoch,
            contrastive_start_epoch=self.contrastive_start_epoch,
            llm_start_epoch=self.llm_start_epoch,
            encoder_lr=self.encoder_lr,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
            val_num_beams=self.val_num_beams,
            length_penalty=self.length_penalty,
            label_smoothing=self.label_smoothing,
            meg_dropout_rate=self.meg_dropout_rate,
            ctc_dropout_rate=self.ctc_dropout_rate,
            optimizer_config=self.optimizer_config,
            scheduler_config=self.scheduler_config,
            preprocess_config=self.preprocess_config,
            llm_metrics=llm_metrics,
            save_dir=self.output_dir,
        )

        # materialise lazy params (channel merger) before DDP wraps the model
        sample = loaders["train"].dataset[0]
        module.eval()
        with torch.no_grad():
            module.network(
                sample.data["neuros"].transpose(1, 2),
                torch.zeros(1, dtype=torch.long),
                sample.data.get("chan_pos"),
            )
        module.train()
        for mod in module.modules():
            for pname, p in list(getattr(mod, "_parameters", {}).items()):
                if isinstance(p, torch.nn.UninitializedParameter):
                    mod._parameters[pname] = nn.Parameter(torch.empty(1))
        return module

    def _trainer_setup(self) -> pl.Trainer:
        accel, devices = accelerator(self.devices)
        if self.eval_only:
            # Evaluate in a single process (like V1): the prediction callback then
            # captures the whole test split with no DDP sharding to reconcile.
            devices = 1
        callbacks: list[pl.Callback] = [PredictionCSVCallback(save_dir=self.output_dir)]
        if self.save_checkpoints:
            # Two checkpoints: the CTC encoder (cer_epo) and the LLM decoder (WER),
            # so the best weights are kept whichever phase you stop in.
            callbacks += [
                ModelCheckpoint(
                    dirpath=self.output_dir,
                    filename="best_ctc",
                    save_last=True,
                    save_top_k=1,
                    monitor="val/cer_epo",
                    mode="min",
                ),
                ModelCheckpoint(
                    dirpath=self.output_dir,
                    filename="best_llm",
                    save_top_k=1,
                    monitor="val/WER",
                    mode="min",
                ),
            ]
        loggers: list = [CSVLogger(self.output_dir, name="logs")]
        if self.wandb_config is not None:
            loggers.append(self._build_wandb_logger())
        return pl.Trainer(
            accelerator=accel,
            devices=devices,
            strategy=DDPStrategy(find_unused_parameters=True) if devices > 1 else "auto",
            max_epochs=self.max_epochs,
            gradient_clip_val=self.gradient_clip_val,
            accumulate_grad_batches=self.accumulate_gradient_batches,
            precision=self.precision,
            callbacks=callbacks,
            logger=loggers,
            log_every_n_steps=2,
        )

    def _build_wandb_logger(self):
        try:
            xp_config = self.model_dump(mode="json")
        except Exception:
            xp_config = None
        return self.wandb_config.build(save_dir=self.output_dir, xp_config=xp_config)

    def run(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        loaders = self.data.build()
        self._module = self._build_module(loaders)
        self._trainer = self._trainer_setup()
        if not self.eval_only:
            self._trainer.fit(
                self._module,
                loaders["train"],
                loaders.get("val"),
                ckpt_path=self.resume_ckpt,
            )
        if "test" in loaders:
            self._trainer.test(
                self._module,
                dataloaders=loaders["test"],
                ckpt_path=self.ckpt_path if self.eval_only else None,
            )


def main(argv: list[str] | None = None) -> None:
    """Run the experiment in a given mode:

    ``python -m brain2qwerty_v2.main {debug,train,eval,cache}`` — every command is
    the same ``Experiment``, only the config (and eval/cache mode) differs.
    """
    import argparse

    import studies  # noqa: F401  (registers the EnglishBCBL study)

    from .cli import add_wandb_args, wandb_config
    from .config.xp_config import debug_config, experiment_config

    parser = argparse.ArgumentParser(prog="brain2qwerty_v2")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("debug", help="1-timeline smoke test (default debug config)")
    p_train = sub.add_parser("train", help="full training (1 node, 8 GPUs)")
    p_train.add_argument("--resume", default=None, help="checkpoint to resume from")
    p_train.add_argument("--seed", type=int, default=None, help="override the seed")
    p_eval = sub.add_parser("eval", help="evaluate a checkpoint on the test split")
    p_eval.add_argument("--ckpt", required=True, help="checkpoint to evaluate")
    p_cache = sub.add_parser("cache", help="pre-warm the feature cache")
    p_cache.add_argument("--debug", action="store_true", help="only the debug subset")
    for p in (sub.choices["debug"], p_train, p_eval):
        add_wandb_args(p)
    args = parser.parse_args(argv)

    if args.command == "cache":
        cfg = debug_config() if args.debug else experiment_config()
        print("[brain2qwerty_v2] pre-warming the feature cache...")
        Experiment(**cfg).data.build()
        print("[brain2qwerty_v2] cache warmed.")
        return

    cfg = debug_config() if args.command == "debug" else experiment_config()
    if args.command == "eval":
        cfg["eval_only"] = True
        cfg["ckpt_path"] = args.ckpt
    if getattr(args, "resume", None):
        cfg["resume_ckpt"] = args.resume
    if getattr(args, "seed", None) is not None:
        cfg["seed"] = args.seed
    wandb = wandb_config(args, args.command, cfg.get("seed", 0))
    if wandb is not None:
        cfg["wandb_config"] = wandb
    print(f"[brain2qwerty_v2] running in '{args.command}' mode (seed={cfg.get('seed')})")
    Experiment(**cfg).run()


if __name__ == "__main__":
    main()
