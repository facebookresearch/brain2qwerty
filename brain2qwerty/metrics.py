# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import Levenshtein
import torch
import torchmetrics

from .utils import CHAR_INDEX


class SentenceCER(torchmetrics.Metric):
    """Online character error rate aggregated across sentences.

    Accumulates per-keystroke predictions, reconstructs character sequences,
    and computes the normalized Levenshtein distance averaged over sentences.
    """

    def __init__(self):
        super().__init__()
        self.add_state(
            "total_distance",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total_length",
            default=torch.tensor(0),
            dist_reduce_fx="sum",
        )

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        _, predicted = torch.max(y_pred, 1)
        pred_str = "".join(CHAR_INDEX.get(idx.item(), "") for idx in predicted)
        true_str = "".join(CHAR_INDEX.get(idx.item(), "") for idx in y_true)
        distance = Levenshtein.distance(pred_str, true_str)
        self.total_distance += distance / max(len(true_str), 1)
        self.total_length += 1

    def compute(self) -> torch.Tensor:
        if self.total_length == 0:
            return torch.tensor(0.0)
        return self.total_distance / self.total_length
