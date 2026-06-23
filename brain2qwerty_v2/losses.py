# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import dtw_matched_pairs


class CtcLoss:
    """Character-level CTC loss on the encoder logits (blank=0)."""

    def __init__(self) -> None:
        self.loss = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    def __call__(self, phoneme_pred, phoneme_true, x_len_norm, y_len):
        log_probs = phoneme_pred.log_softmax(2).permute(1, 0, 2)
        return self.loss(log_probs, phoneme_true, x_len_norm, y_len)


class WordContrastiveLoss(nn.Module):
    """Word-level SigLIP contrastive loss between neural and text word embeddings.

    For each sentence, predicted (segmented) word embeddings are matched 1:1 to the
    ground-truth word embeddings via hard DTW; matched pairs across the batch form a
    SigLIP loss. Near-duplicate ground-truth words (cosine >= ``identical_candidates_
    threshold``) are treated as extra positives to avoid false negatives.
    """

    def __init__(
        self,
        identical_candidates_threshold: float = 0.999,
        reweigh_positives: bool = True,
    ):
        super().__init__()
        self.identical_candidates_threshold = identical_candidates_threshold
        self.reweigh_positives = reweigh_positives
        self.log_temperature = nn.Parameter(torch.tensor(10.0).log())
        self.bias = nn.Parameter(torch.tensor(-10.0))

    def _siglip_contrastive(
        self, pred_mat: torch.Tensor, gt_mat: torch.Tensor
    ) -> torch.Tensor:
        logits = self.log_temperature.exp() * (pred_mat @ gt_mat.T) + self.bias
        gt_sim = gt_mat @ gt_mat.T
        targets = (gt_sim >= self.identical_candidates_threshold).float()
        weights = None
        if self.reweigh_positives:
            weights = 1.0 - targets + torch.eye(targets.shape[0], device=targets.device)
        return (
            F.binary_cross_entropy_with_logits(
                logits, targets, weight=weights, reduction="sum"
            )
            / logits.shape[0]
        )

    def forward(
        self, pred_words: list[torch.Tensor], gt_words: list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        device = pred_words[0].device
        all_pred: list[torch.Tensor] = []
        all_gt: list[torch.Tensor] = []

        for pred, gt in zip(pred_words, gt_words):
            if pred.shape[0] == 0 or gt.shape[0] == 0:
                continue
            pairs = dtw_matched_pairs(pred.detach(), gt.detach())
            if not pairs:
                continue
            pred_idx, gt_idx = zip(*pairs)
            all_pred.append(pred[list(pred_idx)])
            all_gt.append(gt[list(gt_idx)])

        if not all_pred:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return {"loss": zero, "contrastive": zero}

        pred_mat = F.normalize(torch.cat(all_pred), p=2, dim=-1)
        gt_mat = F.normalize(torch.cat(all_gt).detach(), p=2, dim=-1)
        contrastive = self._siglip_contrastive(pred_mat, gt_mat)
        return {"loss": contrastive, "contrastive": contrastive}
