# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Default configuration for Brain2Qwerty on Pinet2024Meg.

Run directly for a quick debug check (2 epochs, 1 subject):

    python -m brain2qwerty.grids.defaults
"""

import os

from ..utils import BUTTON_MAPPING, NUM_CLASSES

PROJECT_NAME = "brain2qwerty"
_BRAINAI_ROOT = os.environ.get("BRAINAI_ROOT", os.path.expanduser("~/brainai"))
_DATA_ROOT = os.environ.get("BRAINAI_DATA_ROOT", os.path.join(_BRAINAI_ROOT, "data"))
CACHE = os.environ.get(
    "BRAINAI_CACHE", os.path.join(_BRAINAI_ROOT, "cache", PROJECT_NAME)
)
SAVEDIR = os.environ.get(
    "BRAINAI_RESULTS", os.path.join(_BRAINAI_ROOT, "results", PROJECT_NAME)
)


default_config = {
    "infra": {
        "cluster": None,
        "folder": SAVEDIR,
        "gpus_per_node": 1,
    },
    "data": {
        "study": {
            "name": "Pinet2024Meg",
            "infra": {"folder": CACHE},
            "path": os.environ.get(
                "BRAINAI_STUDIES_PATH",
                os.path.join(_DATA_ROOT, "studies"),
            ),
            "query": "subject.isin(['Pinet2024Meg/S1'])",
        },
        "neuro": {
            "name": "Meg",
            "frequency": 50,
            "filter": (0.1, 20.0),
            "baseline": (0.0, 0.2),
            "apply_proj": False,
            "clamp": 5,
            "scaler": "RobustScaler",
            "infra": {"folder": CACHE, "cluster": None},
        },
        "feature": {
            "name": "LabelEncoder",
            "aggregation": "trigger",
            "predefined_mapping": BUTTON_MAPPING,
            "event_types": "Button",
            "event_field": "button",
            "return_one_hot": False,
        },
        "num_classes": NUM_CLASSES,
        "start": -0.2,
        "duration": 0.5,
        "batch_size": 128,
        "splitting_seed": 1,
    },
    "brain_model_config": {
        "name": "SimpleConvTimeAgg",
        "time_agg_out": "att",
        "dropout_input": 0.2,
        "conv_dropout": 0.5,
        "hidden": 2048,
        "batch_norm": True,
        "depth": 8,
        "dilation_period": 3,
        "kernel_size": 3,
        "relu_leakiness": 0.01,
        "initial_linear": 512,
        "gelu": True,
        "skip": True,
        "scale": 0.1,
        "subject_layers_config": {},
        "merger_config": {
            "n_virtual_channels": 270,
            "fourier_emb_config": {
                "n_freqs": None,
                "total_dim": 2048,
                "n_dims": 2,
            },
            "dropout": 0.2,
            "usage_penalty": 1.0,
            "per_subject": True,
            "embed_ref": False,
        },
    },
    "transformer_config": {
        "name": "TransformerEncoder",
        "alibi_pos_bias": True,
        "depth": 4,
        "heads": 2,
    },
    "metrics": [
        {
            "log_name": "acc_macro",
            "name": "Accuracy",
            "kwargs": {
                "task": "multiclass",
                "average": "macro",
                "num_classes": NUM_CLASSES,
            },
        },
    ],
    "loss": {"name": "CrossEntropyLoss", "kwargs": {}},
    "n_epochs": 2,
    "save_checkpoints": False,
    "optimizer": {
        "optimizer": {
            "name": "AdamW",
            "lr": 1e-4,
            "kwargs": {"weight_decay": 1e-4},
        },
        "scheduler": {
            "name": "OneCycleLR",
            "kwargs": {
                "max_lr": 1e-4,
                "pct_start": 0.1,
                "total_steps": 100,
            },
        },
    },
}


if __name__ == "__main__":
    from ..main import Experiment

    exp = Experiment(**default_config)
    exp.infra.clear_job()
    exp.run()
