# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Integration smoke test: instantiate the full model from config,
run one forward + backward pass on synthetic data."""

import torch
from torch import nn
from torchmetrics import Accuracy

from brain2qwerty.metrics import SentenceCER
from brain2qwerty.pl_module import BrainModule
from brain2qwerty.utils import NUM_CLASSES

from .conftest import make_fake_batch


def test_full_forward_backward(brain_model, transformer_model):
    """Verify one train step runs without error and produces a gradient."""
    module = BrainModule(
        model=brain_model,
        transformer=transformer_model,
        loss=nn.CrossEntropyLoss(),
        metrics={
            "acc": Accuracy(task="multiclass", num_classes=NUM_CLASSES),
            "CER": SentenceCER(),
        },
        lr=1e-3,
        max_epochs=1,
    )
    module.train()
    batch = make_fake_batch(n_samples=8, n_sentences=2)
    loss = module.training_step(batch, batch_idx=0)
    loss.backward()

    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
    )
    assert has_grad, "No gradients after backward pass"


def test_model_from_config():
    """Verify neuraltrain config can build the architecture used in defaults.py."""
    from neuraltrain.models.simpleconv import SimpleConvTimeAggConfig

    from .conftest import SMALL_CHANNELS, SMALL_FOURIER_DIM, SMALL_HIDDEN

    config = SimpleConvTimeAggConfig(
        name="SimpleConvTimeAgg",
        time_agg_out="att",
        dropout_input=0.0,
        conv_dropout=0.0,
        hidden=SMALL_HIDDEN,
        batch_norm=True,
        depth=2,
        dilation_period=1,
        kernel_size=3,
        relu_leakiness=0.01,
        initial_linear=64,
        gelu=True,
        skip=True,
        scale=0.1,
        subject_layers_config={},
        merger_config={
            "n_virtual_channels": 16,
            "fourier_emb_config": {
                "n_freqs": None,
                "total_dim": SMALL_FOURIER_DIM,
                "n_dims": 2,
            },
            "dropout": 0.0,
            "usage_penalty": 1.0,
            "per_subject": True,
            "embed_ref": False,
        },
    )
    model = config.build(n_in_channels=SMALL_CHANNELS, n_outputs=SMALL_HIDDEN)
    assert hasattr(model, "out_channels")

    x = torch.randn(4, SMALL_CHANNELS, 25)
    subject_ids = torch.zeros(4, dtype=torch.long)
    channel_pos = torch.randn(4, SMALL_CHANNELS, 2)
    out = model(x, subject_ids, channel_pos)
    assert out.shape == (4, SMALL_HIDDEN)
