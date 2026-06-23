# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from .utils import build_mlp, letters_withblank

SPACE_IDX = letters_withblank.index("&")


class _IntraWordMLPPooler(nn.Module):
    """Apply an MLP per frame, then mean-pool over a word's frames."""

    def __init__(self, dim: int, n_layers: int = 2):
        super().__init__()
        self.mlp = build_mlp(dim, dim, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).mean(dim=0)


def build_intra_word_pooler(dim: int, n_layers: int = 2) -> nn.Module:
    return _IntraWordMLPPooler(dim, n_layers)


class CTCSpaceSegmenter(nn.Module):
    """Group encoder frames into pseudo-word embeddings using CTC space predictions.

    Frames between predicted space symbols form a word; each word's frames are
    pooled by the learned intra-word pooler. Returns one ``(N_words_i, D)`` tensor
    per sample.
    """

    def __init__(
        self,
        include_blanks: bool = True,
        min_word_frames: int = 1,
        intra_word_pooler: nn.Module | None = None,
    ):
        super().__init__()
        self.include_blanks = include_blanks
        self.min_word_frames = min_word_frames
        self.intra_word_pooler = intra_word_pooler

    def _pool(self, frames_tensor: torch.Tensor) -> torch.Tensor:
        if self.intra_word_pooler is not None:
            return self.intra_word_pooler(frames_tensor)
        return frames_tensor.mean(dim=0)

    def forward(
        self, z_final: torch.Tensor, ctc_logits: torch.Tensor
    ) -> list[torch.Tensor]:
        with torch.no_grad():
            preds = ctc_logits.argmax(dim=-1)
        B, T, _ = z_final.shape
        results: list[torch.Tensor] = []
        for b in range(B):
            is_space = preds[b] == SPACE_IDX
            word_segments: list[list[int]] = []
            current: list[int] = []
            for t in range(T):
                if is_space[t]:
                    if current:
                        word_segments.append(current)
                        current = []
                elif self.include_blanks or preds[b, t] != 0:
                    current.append(t)
            if current:
                word_segments.append(current)
            word_segments = [s for s in word_segments if len(s) >= self.min_word_frames]

            if not word_segments:
                results.append(self._pool(z_final[b]).unsqueeze(0))
            else:
                embeds = [
                    self._pool(
                        z_final[b].index_select(0, torch.tensor(s, device=z_final.device))
                    )
                    for s in word_segments
                ]
                results.append(torch.stack(embeds))
        return results
