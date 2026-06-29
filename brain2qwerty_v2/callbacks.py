# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
from pathlib import Path

import lightning.pytorch as pl

from .utils import compute_sample_metrics, prediction_fieldnames


class PredictionCSVCallback(pl.Callback):
    """Save per-sentence predictions (text + CER/WER/SemER) to ``predictions_test.csv``.

    Only written at test time (not every validation epoch). The rows accumulated by
    the Lightning module are gathered across ranks first, which is a no-op for the
    single-process ``eval`` and keeps the training-end test complete on multi-GPU.
    """

    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = Path(save_dir)

    @staticmethod
    def _gather_rows(trainer, rows: list[dict]) -> list[dict]:
        if trainer.world_size <= 1:
            return rows
        import torch.distributed as dist

        gathered: list[list[dict] | None] = [None] * trainer.world_size
        dist.all_gather_object(gathered, rows)
        if trainer.global_rank == 0:
            return [r for rank_rows in gathered for r in rank_rows]
        return []

    def _save(self, trainer, rows, filename, with_semer):
        rows = self._gather_rows(trainer, rows)
        if not rows or trainer.global_rank != 0:
            return
        ctc_texts = [r.get("ctc_text", "") for r in rows]
        has_ctc = any(ctc_texts)
        has_segment_meta = any(r.get("subject") for r in rows)
        rows_with_metrics = compute_sample_metrics(
            [r["true_text"] for r in rows],
            [r["pred_text"] for r in rows],
            ctc_texts=ctc_texts if has_ctc else None,
            with_semer=with_semer,
        )
        for row_m, row_raw in zip(rows_with_metrics, rows):
            if has_segment_meta:
                row_m["subject"] = row_raw.get("subject", "")
                row_m["sentence_UID"] = row_raw.get("sentence_UID", "")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / filename
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=prediction_fieldnames(has_ctc, has_segment_meta)
            )
            writer.writeheader()
            writer.writerows(rows_with_metrics)
        print(f"Saved {len(rows_with_metrics)} predictions to {path}")

    def on_test_epoch_end(self, trainer, pl_module):
        rows = getattr(pl_module, "_test_predictions", [])
        self._save(trainer, rows, "predictions_test.csv", with_semer=True)
