# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
from types import SimpleNamespace

import torch

from brain2qwerty_v1.config.model_config import ENCODER, TRANSFORMER
from brain2qwerty_v1.metrics import CER
from brain2qwerty_v1.pl_module import BrainModule
from neuraltrain.models.base import BaseModelConfig
from neuraltrain.optimizers import LightningOptimizer

N_CH = 306
N_CLASSES = 29
HIDDEN = 32


def _tiny_module() -> BrainModule:
    enc_cfg = copy.deepcopy(ENCODER)
    enc_cfg.update(hidden=HIDDEN, depth=2, initial_linear=16)
    enc_cfg["merger_config"].update(n_virtual_channels=16)
    enc_cfg["merger_config"]["fourier_emb_config"].update(total_dim=HIDDEN)
    encoder = BaseModelConfig(**enc_cfg).build(n_in_channels=N_CH, n_outputs=HIDDEN)

    tr_cfg = copy.deepcopy(TRANSFORMER)
    tr_cfg.update(depth=1, heads=1)
    transformer = BaseModelConfig(**tr_cfg).build(dim=HIDDEN)

    optimizer = LightningOptimizer(
        optimizer={"name": "AdamW", "lr": 5e-5},
        scheduler={"name": "OneCycleLR", "kwargs": {"max_lr": 5e-5}},
    )
    return BrainModule(
        model=encoder,
        transformer=transformer,
        loss=torch.nn.CrossEntropyLoss(),
        metrics={"CER": CER()},
        optimizer=optimizer,
    )


def _fake_batch(n: int = 4):
    # 4 keystrokes grouped into 2 sentences (A, A, B, B). The sentence_UID on each
    # segment is what BrainModule uses to regroup per-keystroke embeddings into
    # sentences before the transformer, so the fake batch must carry it.
    uids = ["A", "A", "B", "B"][:n]
    data = {
        "neuro": torch.randn(n, N_CH, 25),
        "subject_id": torch.zeros(n, 1, dtype=torch.long),
        "channel_positions": torch.rand(n, N_CH, 2),
        "feature": torch.randint(0, N_CLASSES, (n, 1)),
    }
    segments = [
        SimpleNamespace(trigger=SimpleNamespace(extra={"sentence_UID": uid}))
        for uid in uids
    ]
    return SimpleNamespace(data=data, segments=segments)


def test_brain_module_forward_and_backward():
    """Exercise the full BrainModule step on a synthetic batch.

    Why it matters: this is the integration point between data layout and model.
    It verifies the two-stage forward (per-keystroke encoder embeddings ->
    sentence-grouped transformer logits) yields the right shapes and that a
    backward pass produces finite gradients across the whole module — the cheapest
    way to catch a broken training step without a GPU or real data.
    """
    torch.manual_seed(0)
    module = _tiny_module()
    batch = _fake_batch()

    emb = module.forward(batch)
    assert emb.shape == (4, HIDDEN)  # one embedding per keystroke

    logits = module._transformer_forward(batch, emb)
    assert logits.shape == (4, N_CLASSES)  # per-keystroke character logits

    loss = module.loss(logits, batch.data["feature"].squeeze(1))
    loss.backward()
    assert loss.item() > 0
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    assert grads
    assert all(not torch.isnan(g).any() for g in grads)
