# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Fast (synthetic, CPU) tests for the V2 losses and CTC decoding."""

import torch

from brain2qwerty_v2.losses import CtcLoss, WordContrastiveLoss
from brain2qwerty_v2.utils import (
    ctc_greedy_decode,
    dtw_matched_pairs,
    hard_dtw_path,
    label_to_text,
)


def test_ctc_loss_backward():
    # The character-level CTC loss (stage-1 objective) must be finite and produce
    # clean gradients for variable-length targets — the basic guarantee that the
    # encoder can be trained at all.
    torch.manual_seed(0)
    B, T, C = 2, 20, 29
    logits = torch.randn(B, T, C, requires_grad=True)
    targets = torch.randint(1, C, (B, 5))  # class 0 is the CTC blank
    in_lens = torch.full((B,), T, dtype=torch.long)
    tgt_lens = torch.full((B,), 5, dtype=torch.long)
    loss = CtcLoss()(logits, targets, in_lens, tgt_lens)
    loss.backward()
    assert loss.item() > 0 and logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_hard_dtw_monotonic_path():
    """The DTW alignment used by the contrastive loss is well-formed.

    Why it matters: the word-level contrastive objective aligns predicted word
    embeddings to ground-truth ones via DTW. The path must start at (0,0), end at
    the bottom-right, and be monotonic in both axes (a valid warping); otherwise
    the matched pairs fed to the contrastive loss would be garbage. We also check
    each predicted index is matched to a single ground-truth index.
    """
    cost = torch.rand(4, 6)
    path = hard_dtw_path(cost)
    assert path[0] == (0, 0) and path[-1] == (3, 5)
    # monotonic non-decreasing in both indices (a valid alignment)
    for (a0, b0), (a1, b1) in zip(path, path[1:]):
        assert a1 >= a0 and b1 >= b0
    pairs = dtw_matched_pairs(torch.randn(4, 8), torch.randn(6, 8))
    assert len({p for p, _ in pairs}) == len(pairs)  # one gt per pred


def test_word_contrastive_loss_backward():
    """The SigLIP-style word contrastive loss is non-trivial and differentiable.

    Why it matters: this is stage-2's objective, aligning per-sentence lists of
    predicted word vectors to the LLM target embeddings. We run it with
    dtw_weight=0 (pure contrastive term) and require a non-zero, NaN-free
    gradient so it can actually pull embeddings together during training.
    """
    torch.manual_seed(0)
    D = 16
    # ragged batch: 2 sentences with 3 and 2 words respectively
    pred = [torch.randn(3, D, requires_grad=True), torch.randn(2, D, requires_grad=True)]
    gt = [torch.randn(3, D), torch.randn(2, D)]
    loss_fn = WordContrastiveLoss()
    out = loss_fn(pred, gt)
    out["loss"].backward()
    assert out["loss"].item() != 0.0
    assert pred[0].grad is not None and not torch.isnan(pred[0].grad).any()


def test_ctc_greedy_decode_and_label_to_text():
    # CTC greedy decoding semantics: collapse repeats and drop the blank (class 0),
    # then map indices to characters. The crafted logits spell [1,2,3] after
    # collapsing the interleaved blanks; label 27 ("&") must render as a space.
    B, T, C = 1, 10, 29
    logits = torch.full((B, T, C), -10.0)
    seq = [1, 0, 2, 0, 3, 0, 0, 0, 0, 0]  # collapses to [1, 2, 3] -> "abc"
    for t, c in enumerate(seq):
        logits[0, t, c] = 10.0
    assert ctc_greedy_decode(logits) == ["abc"]
    assert label_to_text([1, 2, 3]) == "abc"
    assert label_to_text([1, 27, 2]) == "a b"  # 27 -> "&" -> space
