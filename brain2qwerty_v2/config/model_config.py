# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Convolutional encoder with a per-subject 2D-Fourier channel merger.
_ENCODER = {
    "name": "SimpleConv",
    "dropout_input": 0.2,
    "conv_dropout": 0.5,
    "hidden": 1500,
    "batch_norm": True,
    "depth": 4,
    "dilation_period": 3,
    "kernel_size": 5,
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

# Full encoder: conv encoder -> temporal downsampling -> Conformer, with an
# auxiliary CTC head (z_aux) blended back into the transformer input.
ENCODER = {
    "name": "ConvConformer",
    "dim": 1024,
    "encoder_config": {**_ENCODER},
    "transformer_config": {
        "name": "Conformer",
        "ffn_dim": 1024,
        "num_heads": 4,
        "num_layers": 4,
        "depthwise_conv_kernel_size": 17,
        "dropout": 0.3,
        "use_group_norm": True,
        "convolution_first": False,
    },
    "temporal_downsampling_config": {"kernel_size": 16, "stride": 4},
    "aux_prediction": True,
}
