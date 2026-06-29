# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

import torch
from lightning.pytorch.callbacks import Callback


class LogSentencePredictions(Callback):
    """Save per-sentence predictions as JSON at test time.

    For every sentence it stores the predicted class indices, the typed
    (ground-truth) class indices, the sentence text, the subject ID and the
    timeline (via ``sentence_UID``) and the raw logits, so the output can be
    post-processed (e.g. ``scripts/extract_predictions.py`` and
    ``scripts/ngram_decoding.py``). Only saved at test (not validation) to keep
    training epochs fast.
    """

    def __init__(self, save_dir: str | None = None):
        super().__init__()
        self.save_dir = save_dir

    def _collect_batch(self, outputs, batch, store):
        y_pred, y_true = outputs
        for pred, true, seg in zip(y_pred, y_true, batch.segments):
            uid = seg.trigger.extra.get("sentence_UID")
            sentence = getattr(seg.trigger, "sentence", None) or ""
            _, predicted_index = torch.max(pred, dim=0)
            if uid not in store:
                store[uid] = {"pred": [], "typed": [], "true": sentence, "logits": []}
            store[uid]["pred"].append(predicted_index.item())
            store[uid]["typed"].append(true.item())
            store[uid]["logits"].append(pred.tolist())

    def _resolve_dir(self, trainer):
        base = self.save_dir
        if base is None and trainer.logger is not None:
            base = trainer.logger.save_dir
        return os.path.join(base or ".", "callbacks")

    def _save(self, trainer, store, split):
        if trainer.global_rank != 0:
            return
        save_dir = self._resolve_dir(trainer)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{split}_all_sentences.json"), "w") as f:
            json.dump(store, f, indent=4)

    def on_test_epoch_start(self, trainer, pl_module):
        self._test_store: dict = {}

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._collect_batch(outputs, batch, self._test_store)

    def on_test_epoch_end(self, trainer, pl_module):
        self._save(trainer, self._test_store, "test")
