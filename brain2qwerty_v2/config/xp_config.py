# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

from .model_config import ENCODER

STUDY_PATH = os.environ.get(
    "BRAIN2QWERTY_STUDIES_EN", str(Path.home() / "brain2qwerty_data" / "pinet2025")
)
CACHE = os.environ.get("BRAIN2QWERTY_CACHE", str(Path.home() / ".cache" / "brain2qwerty"))
RESULTS = os.environ.get("BRAIN2QWERTY_RESULTS", str(Path(CACHE) / "results"))

# the paper uses a different LLM for the reported performance
LLM = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Frozen LLM token embeddings used as the word-level contrastive target.
WORD_EXTRACTOR = {"model_name": LLM, "layers": 0, "contextualized": False}


def experiment_config() -> dict:
    """Full Brain2Qwerty V2 configuration (EnglishBCBL, MEG; CTC + contrastive + LLM)."""
    return {
        "output_dir": RESULTS,
        "seed": 123,
        "max_epochs": 275,
        "data": {
            "study": {
                "name": "PinetAudio2025",
                "path": STUDY_PATH,
                "infra": {"folder": CACHE},
                "infra_timelines": {"folder": CACHE, "cluster": None},
            },
            "transforms": [
                {"name": "EnglishBCBLPreprocessing"},
                {
                    "name": "Brain2QwertyV2Splitter",
                    "deterministic_splitter": {
                        "ratios": {"train": 0.8, "val": 0.1, "test": 0.1}
                    },
                },
                {"name": "WordCreator"},
            ],
            "neuro": {
                "name": "MegExtractor",
                "frequency": 100,
                "filter": (0.5, 45.0),
                "scaler": "RobustScaler",
                "apply_proj": False,
                "clamp": 5,
                "picks": "meg",
                "notch_filter": 50,
                "allow_maxshield": True,
                "infra": {"folder": CACHE, "cluster": None},
            },
            "extractor": {
                "name": "SentenceKeySeq",
                "mode": "typed_label",
                "infra": {"folder": CACHE},
            },
            "batch_size": 64,
            "val_batch_size": 128,
            "test_batch_size": 8,
            "num_workers": 16,
            "pin_memory": True,
            "persistent_workers": True,
        },
        # MEG augmentation (on-device, train only): per-channel offset + SpecAugment
        # masking + time-stretch (no white noise), matching the paper.
        "preprocess_config": {
            "whiteNoiseSD": 0.0,
            "constantOffsetSD": 0.3,
            "time_mask_param": 50,
            "p_time_mask": 0.2,
            "freq_mask_param": 400,
            "time_stretch": True,
        },
        "brain_model_config": ENCODER,
        # staged 3-loss schedule: CTC from 0, +contrastive at 150, +LLM at 225
        "alpha": 0.1,
        "beta": 0.01,
        "loss_alpha": 0.7,
        "ctc_start_epoch": 0,
        "contrastive_start_epoch": 150,
        "llm_start_epoch": 225,
        # LLM + LoRA all-subjects rank=2
        "llm_name": LLM,
        "lora_rank": 2,
        "word_extractor_config": WORD_EXTRACTOR,
        "num_beams": 16,
        "optimizer_config": {"lr": 8e-4, "weight_decay": 1e-3},
        "scheduler_config": {
            "name": "WarmupCosine",
            "warmup_steps": 500,
            "eta_min": 1e-6,
        },
        "accumulate_gradient_batches": 2,
        "precision": "bf16-mixed",
    }


def debug_config() -> dict:
    """Smoke-test config: one timeline, all losses from epoch 0, single GPU."""
    cfg = experiment_config()
    cfg["data"]["study"]["query"] = "timeline_index == 0"
    cfg["data"]["batch_size"] = 4
    cfg["data"]["val_batch_size"] = 4
    cfg["data"]["test_batch_size"] = 4
    cfg["max_epochs"] = 2
    cfg["contrastive_start_epoch"] = 0
    cfg["llm_start_epoch"] = 0
    cfg["num_beams"] = 1
    cfg["devices"] = 1
    cfg["save_checkpoints"] = False
    return cfg
