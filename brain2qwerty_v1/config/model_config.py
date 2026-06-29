# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Convolutional encoder with a per-subject 2D-Fourier channel merger; produces
# one embedding per keystroke window.
ENCODER = {
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
        "fourier_emb_config": {"n_freqs": None, "total_dim": 2048, "n_dims": 2},
        "dropout": 0.2,
        "usage_penalty": 1.0,
        "per_subject": True,
        "embed_ref": False,
    },
}

# Sentence-level transformer over the per-keystroke embeddings.
TRANSFORMER = {
    "name": "TransformerEncoder",
    "alibi_pos_bias": True,
    "depth": 4,
    "heads": 2,
}
