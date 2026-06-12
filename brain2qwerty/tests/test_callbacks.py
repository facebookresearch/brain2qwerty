# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from types import SimpleNamespace

import torch

from brain2qwerty.callbacks import LogSentencePredictions
from brain2qwerty.utils import NUM_CLASSES

from .conftest import make_fake_batch


def test_collect_batch_populates_store():
    cb = LogSentencePredictions()
    batch = make_fake_batch(n_samples=8, n_sentences=2)
    y_pred = torch.randn(8, NUM_CLASSES)
    y_true = torch.randint(0, NUM_CLASSES, (8,))

    store = {}
    cb._collect_batch((y_pred, y_true), batch, store)
    assert len(store) > 0
    for uid, entry in store.items():
        assert "pred" in entry
        assert "typed" in entry
        assert "true" in entry
        assert "logits" in entry
        assert len(entry["pred"]) == len(entry["typed"])


def test_save_writes_json(tmp_path):
    cb = LogSentencePredictions()
    store = {
        "sent_0": {
            "pred": [0, 1, 2],
            "typed": [0, 1, 3],
            "true": "abc",
            "logits": [[0.1] * NUM_CLASSES] * 3,
        }
    }
    mock_trainer = SimpleNamespace(logger=SimpleNamespace(save_dir=str(tmp_path)))
    cb._save(mock_trainer, store, "test")

    out_path = tmp_path / "callbacks" / "test_all_sentences.json"
    assert out_path.exists()
    with open(out_path) as f:
        data = json.load(f)
    assert "sent_0" in data
    assert data["sent_0"]["true"] == "abc"
