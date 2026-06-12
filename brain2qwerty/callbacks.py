# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback


class LogSentencePredictions(Callback):
    """Save per-sentence predictions as JSON at the end of each val/test epoch.

    For every sentence, stores the predicted class indices, the typed (ground-truth)
    class indices, the original sentence text, and the raw logits. The output JSON
    can be post-processed with ``scripts/extract_predictions.py``.
    """

    def _collect_batch(self, outputs, batch, store):
        y_pred, y_true = outputs
        sentence_uids = np.array(
            [
                f"{seg._trigger['trial_id']}_{seg._trigger['timeline']}"
                for seg in batch.segments
            ]
        )
        true_sentences = [seg._trigger["sentence"] for seg in batch.segments]

        for pred, true, uid, sentence in zip(
            y_pred, y_true, sentence_uids, true_sentences
        ):
            _, predicted_index = torch.max(pred, dim=0)

            if uid not in store:
                store[uid] = {
                    "pred": [],
                    "typed": [],
                    "true": "",
                    "logits": [],
                }
            store[uid]["pred"].append(predicted_index.item())
            store[uid]["typed"].append(true.item())
            store[uid]["true"] = sentence
            store[uid]["logits"].append(pred.tolist())

    def _save(self, trainer, store, split):
        save_dir = os.path.join(trainer.logger.save_dir, "callbacks")
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{split}_all_sentences.json")
        with open(path, "w") as f:
            json.dump(store, f, indent=4)

    # --- validation ---

    def on_validation_epoch_start(self, trainer, pl_module):
        self._val_store = {}

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._collect_batch(outputs, batch, self._val_store)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._save(trainer, self._val_store, "val")

    # --- test ---

    def on_test_epoch_start(self, trainer, pl_module):
        self._test_store = {}

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._collect_batch(outputs, batch, self._test_store)

    def on_test_epoch_end(self, trainer, pl_module):
        self._save(trainer, self._test_store, "test")
