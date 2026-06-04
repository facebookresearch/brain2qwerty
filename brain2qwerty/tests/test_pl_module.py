# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torchmetrics import Accuracy

from brain2qwerty.metrics import SentenceCER
from brain2qwerty.pl_module import BrainModule
from brain2qwerty.utils import NUM_CLASSES

from .conftest import SMALL_HIDDEN, make_fake_batch


def _build_module(brain_model, transformer_model):
    return BrainModule(
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


def test_instantiation(brain_model, transformer_model):
    module = _build_module(brain_model, transformer_model)
    assert isinstance(module, BrainModule)
    assert module.transformer is not None
    assert module.linear.out_features == NUM_CLASSES


def test_forward_shape(brain_model, transformer_model):
    module = _build_module(brain_model, transformer_model)
    module.eval()
    batch = make_fake_batch(n_samples=8, n_sentences=2)
    with torch.no_grad():
        out = module.forward(batch)
    assert out.shape == (8, SMALL_HIDDEN)


def test_training_step_returns_loss(brain_model, transformer_model):
    module = _build_module(brain_model, transformer_model)
    module.train()
    batch = make_fake_batch(n_samples=8, n_sentences=2)
    loss = module.training_step(batch, batch_idx=0)
    assert loss.ndim == 0
    assert loss.requires_grad


def test_validation_step_returns_preds(brain_model, transformer_model):
    module = _build_module(brain_model, transformer_model)
    module.eval()
    batch = make_fake_batch(n_samples=8, n_sentences=2)
    with torch.no_grad():
        y_pred, y_true = module.validation_step(batch, batch_idx=0)
    assert y_pred.shape[0] == 8
    assert y_pred.shape[1] == NUM_CLASSES
    assert y_true.shape[0] == 8
