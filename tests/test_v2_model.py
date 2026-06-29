# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Fast (synthetic, CPU) tests for the V2 encoder, segmenter and CTC decode."""

import copy

import torch

import brain2qwerty_v2.models  # noqa: F401  (registers the ConvConformer encoder)
from brain2qwerty_v2.config.model_config import ENCODER
from brain2qwerty_v2.ctc_segmenter import CTCSpaceSegmenter, build_intra_word_pooler
from brain2qwerty_v2.utils import ctc_greedy_decode
from neuraltrain.models.base import BaseModelConfig

N_CH = 306
N_CLASSES = 29
DIM = 32


def _tiny_encoder():
    cfg = copy.deepcopy(ENCODER)
    cfg["dim"] = DIM
    cfg["encoder_config"].update(hidden=64, depth=2, initial_linear=16)
    cfg["encoder_config"]["merger_config"].update(n_virtual_channels=16)
    cfg["encoder_config"]["merger_config"]["fourier_emb_config"].update(total_dim=DIM)
    cfg["transformer_config"].update(ffn_dim=DIM, num_heads=2, num_layers=1)
    return BaseModelConfig(**cfg).build(n_in_channels=N_CH, n_outputs=N_CLASSES)


def test_encoder_forward_returns_zfinal_and_aux():
    """The ConvConformer exposes the extra outputs the V2 pipeline depends on.

    Why it matters: V2 subclasses the public ConvTransformer to re-add two things
    the downstream losses need — ``z_final`` (per-frame features for the word
    segmenter / contrastive branch) and ``z_aux`` (the auxiliary CTC head). This
    test pins their presence, the class dimension of the CTC heads, and that
    ``z_final`` and ``c_out`` share the (downsampled) temporal length, so the
    segmenter can index frames by CTC timestep.
    """
    torch.manual_seed(0)
    enc = _tiny_encoder()
    b, t = 2, 120
    neuros = torch.randn(b, t, N_CH)  # (B, T, C) sentence layout
    days = torch.zeros(b, dtype=torch.long)
    chan_pos = torch.rand(b, N_CH, 2)

    out = enc(neuros, days, chan_pos)
    assert set(("z", "z_enc", "z_final", "c_out", "z_aux")).issubset(out.keys())
    assert out["z_final"].shape[0] == b and out["z_final"].shape[2] == DIM
    assert out["c_out"].shape[2] == N_CLASSES
    # aux CTC head shares the temporal length with c_out
    assert out["z_aux"].shape[2] == N_CLASSES
    assert out["z_final"].shape[1] == out["c_out"].shape[1]


def test_segmenter_and_greedy_decode():
    """The CTC-driven word segmenter produces one pooled vector per word.

    Why it matters: the contrastive stage needs encoder frames grouped into
    pseudo-words using the CTC emissions (space-delimited), each pooled to a single
    D-dim vector to compare against the LLM word embeddings. This checks the
    segmenter returns a per-sentence list of (n_words, D) tensors and that greedy
    decoding yields one string per item.
    """
    torch.manual_seed(0)
    b, t = 2, 40
    z_final = torch.randn(b, t, DIM)
    ctc_logits = torch.randn(b, t, N_CLASSES)

    pooler = build_intra_word_pooler(DIM, n_layers=2)
    seg = CTCSpaceSegmenter(include_blanks=True, intra_word_pooler=pooler)
    words = seg(z_final, ctc_logits)
    assert len(words) == b
    for w in words:
        assert w.ndim == 2 and w.shape[1] == DIM  # (n_words, D) per sentence

    texts = ctc_greedy_decode(ctc_logits)
    assert len(texts) == b and all(isinstance(s, str) for s in texts)
