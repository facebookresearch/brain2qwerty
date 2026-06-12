# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch

kenlm = pytest.importorskip("kenlm")

from brain2qwerty.scripts.ngram_decoding import BeamDecoder, parse_logits  # noqa: E402


@pytest.fixture
def fake_arpa(tmp_path):
    """Create a minimal valid ARPA language model file."""
    chars = list("abcdefghijklmnopqrstuvwxyz&@")
    unigrams = [f"-1.0\t{c}\t-0.5" for c in chars]
    unigrams += ["-99.0\t<unk>\t0.0", "-1.0\t<s>\t-0.5", "-1.0\t</s>\t0.0"]
    bigrams = [f"-0.5\t{a}\t{b}" for a in chars[:5] for b in chars[:5]]
    bigrams += [f"-0.5\t<s>\t{c}" for c in chars[:5]]

    arpa_content = (
        "\\data\\\n"
        f"ngram 1={len(unigrams)}\n"
        f"ngram 2={len(bigrams)}\n\n"
        "\\1-grams:\n" + "\n".join(unigrams) + "\n\n"
        "\\2-grams:\n" + "\n".join(bigrams) + "\n\n"
        "\\end\\\n"
    )
    arpa_path = tmp_path / "fake.arpa"
    arpa_path.write_text(arpa_content)
    return str(arpa_path)


def test_beam_decoder_produces_output(fake_arpa):
    lm = kenlm.Model(fake_arpa)
    decoder = BeamDecoder(lm, beam_size=5, lm_weight=1.0)
    emissions = torch.randn(5, 29)
    result = decoder.decode(emissions)
    assert isinstance(result, str)
    assert len(result) == 5


def test_beam_decoder_strong_signal(fake_arpa):
    """When logits strongly favor specific characters, the decoder should
    output those characters (LM weight is low so brain signal dominates)."""
    lm = kenlm.Model(fake_arpa)
    decoder = BeamDecoder(lm, beam_size=5, lm_weight=0.01)

    from brain2qwerty.scripts.ngram_decoding import ID2CHAR

    target_indices = [0, 1, 2, 3, 7]
    expected = "".join(ID2CHAR[i] for i in target_indices).replace("&", " ")

    emissions = torch.full((5, 29), -10.0)
    for t, idx in enumerate(target_indices):
        emissions[t, idx] = 10.0

    result = decoder.decode(emissions)
    assert result == expected


def test_parse_logits_string():
    s = "[[1.0, 2.0], [3.0, 4.0]]"
    result = parse_logits(s)
    assert result == [[1.0, 2.0], [3.0, 4.0]]


def test_parse_logits_list():
    data = [[1.0, 2.0], [3.0, 4.0]]
    assert parse_logits(data) is data
