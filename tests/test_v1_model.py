# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch

from brain2qwerty_v1.config.model_config import ENCODER, TRANSFORMER
from brain2qwerty_v1.metrics import CER
from neuraltrain.losses.base import CrossEntropyLoss
from neuraltrain.models.base import BaseModelConfig

N_CH = 306
N_CLASSES = 29
HIDDEN = 32  # tiny for speed; the full-size config is validated by the debug run


def _tiny_encoder_config() -> BaseModelConfig:
    cfg = copy.deepcopy(ENCODER)
    cfg.update(hidden=HIDDEN, depth=2, initial_linear=16)
    cfg["merger_config"].update(n_virtual_channels=16)
    cfg["merger_config"]["fourier_emb_config"].update(total_dim=HIDDEN)
    return BaseModelConfig(**cfg)


def _tiny_transformer_config() -> BaseModelConfig:
    cfg = copy.deepcopy(TRANSFORMER)
    cfg.update(depth=1, heads=1)
    return BaseModelConfig(**cfg)


def test_encoder_and_transformer_forward():
    """End-to-end shape check of the V1 forward path on a tiny model.

    Why it matters: this is the contract the LightningModule relies on — the
    convolutional encoder turns each keystroke window into one embedding, those
    embeddings feed the sentence-level transformer, and a linear head produces
    per-character logits. It catches silent breakages in the encoder/transformer
    wiring (built from the public neuraltrain configs) without needing real data.
    """
    torch.manual_seed(0)
    encoder = _tiny_encoder_config().build(n_in_channels=N_CH, n_outputs=HIDDEN)
    transformer = _tiny_transformer_config().build(dim=HIDDEN)
    linear = torch.nn.Linear(HIDDEN, N_CLASSES)

    b = 4  # keystrokes
    enc_out = encoder(
        torch.randn(b, N_CH, 25),
        torch.zeros(b, 1, dtype=torch.long),
        torch.rand(b, N_CH, 2),  # 2D channel positions (per the paper's merger)
    )
    # the encoder may return a dict of intermediate tensors; the keystroke
    # embedding is "c_out" (fall back to the first value if the API changes)
    if isinstance(enc_out, dict):
        enc_out = enc_out.get("c_out", next(iter(enc_out.values())))
    emb = enc_out.reshape(b, -1)
    assert emb.shape == (b, HIDDEN)

    # one sentence of b keystrokes (batch dim 1, all positions valid)
    tr_out = transformer(emb.unsqueeze(0), mask=torch.ones(1, b, dtype=torch.bool))
    logits = linear(tr_out.reshape(b, HIDDEN))
    assert logits.shape == (b, N_CLASSES)


def test_crossentropy_loss_backward():
    """The CE loss (auto-detected from neuraltrain) is finite and differentiable.

    Why it matters: V1 deliberately has no losses.py — training relies on
    neuraltrain resolving CrossEntropyLoss. This guards that the resolved loss is
    usable and produces clean (non-NaN) gradients.
    """
    loss_fn = CrossEntropyLoss().build()
    logits = torch.randn(5, N_CLASSES, requires_grad=True)
    loss = loss_fn(logits, torch.randint(0, N_CLASSES, (5,)))
    loss.backward()
    assert loss.item() > 0
    assert logits.grad is not None and not torch.isnan(logits.grad).any()


def test_cer_metric_perfect_prediction():
    # Sanity for the custom CER metric used to monitor/checkpoint: a perfect
    # one-hot prediction must score exactly 0.
    metric = CER()
    y_true = torch.tensor([0, 1, 2, 3])
    y_pred = torch.nn.functional.one_hot(y_true, N_CLASSES).float()
    metric.update(y_pred, y_true)
    assert float(metric.compute()) == 0.0
