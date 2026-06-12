#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Apply N-gram language model post-processing to model predictions.

Reads a predictions CSV (as produced by extract_predictions.py)
containing per-sentence logits, applies character-level beam search
with a KenLM N-gram model, and outputs an updated CSV with refined
predictions and recomputed CER/WER.

Usage::

    python ngram_decoding.py \\
        --input predictions.csv \\
        --lm news_9gram.arpa \\
        --output predictions_lm.csv
"""

import argparse
import ast

import kenlm
import Levenshtein
import pandas as pd
import torch

CHAR2ID = {
    "s": 0,
    "o": 1,
    "t": 2,
    "e": 3,
    "n": 4,
    "c": 5,
    "i": 6,
    "a": 7,
    "&": 8,
    "d": 9,
    "l": 10,
    "r": 11,
    "b": 12,
    "@": 13,
    "z": 14,
    "v": 15,
    "f": 16,
    "m": 17,
    "u": 18,
    "h": 19,
    "p": 20,
    "g": 21,
    "q": 22,
    "w": 23,
    "x": 24,
    "y": 25,
    "j": 26,
    "k": 27,
    "9": 28,
}
ID2CHAR = {v: k for k, v in CHAR2ID.items()}


class BeamState:
    def __init__(
        self,
        sentence: str,
        score: float,
        lm_state: kenlm.State = None,
    ):
        self.sentence = sentence
        self.score = score
        self.lm_state = lm_state or kenlm.State()

    def __repr__(self):
        return self.sentence


class BeamDecoder:
    """Character-level beam search decoder with KenLM."""

    def __init__(
        self,
        lm: kenlm.Model,
        beam_size: int = 30,
        max_labels_per_timestep: int = 50,
        lm_weight: float = 5.0,
    ):
        self.lm = lm
        self.beam_size = beam_size
        self.max_labels_per_timestep = max_labels_per_timestep
        self.lm_weight = lm_weight

    def decode(self, emissions: torch.Tensor) -> str:
        beam = [BeamState(sentence="", score=0)]
        self.lm.BeginSentenceWrite(beam[0].lm_state)

        for logits in emissions:
            probs = torch.softmax(logits, dim=0)
            top_idx = probs.argsort(descending=True)[: self.max_labels_per_timestep]

            new_beam = []
            for hyp in beam:
                for idx in top_idx:
                    char = ID2CHAR[idx.item()]
                    if char.isdigit():
                        continue

                    new_state = kenlm.State()
                    lm_score = self.lm.BaseScore(hyp.lm_state, char, new_state)
                    brain_score = torch.log(probs[idx])
                    score = hyp.score + self.lm_weight * lm_score + brain_score

                    new_beam.append(
                        BeamState(
                            sentence=hyp.sentence + char,
                            score=score,
                            lm_state=new_state,
                        )
                    )

            beam = sorted(new_beam, key=lambda x: x.score, reverse=True)[: self.beam_size]

        return beam[0].sentence.replace("&", " ")


def parse_logits(logits_str):
    """Parse a logits column value into a list of lists."""
    if isinstance(logits_str, str):
        return ast.literal_eval(logits_str)
    return logits_str


def main():
    parser = argparse.ArgumentParser(description="N-gram LM post-processing")
    parser.add_argument(
        "--input",
        required=True,
        help="Input predictions CSV",
    )
    parser.add_argument(
        "--lm",
        required=True,
        help="Path to KenLM .arpa language model",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: overwrite input)",
    )
    parser.add_argument(
        "--lm-weight",
        type=float,
        default=5.0,
        help="Language model weight (default: 5.0)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=30,
        help="Beam size (default: 30)",
    )
    args = parser.parse_args()

    print(f"Loading language model: {args.lm}")
    lm = kenlm.Model(args.lm)
    decoder = BeamDecoder(lm, beam_size=args.beam_size, lm_weight=args.lm_weight)

    print(f"Loading predictions: {args.input}")
    df = pd.read_csv(args.input)
    n_subj = df["Subject"].nunique()
    print(f"  {len(df)} sentences, {n_subj} subjects")

    lm_predictions = []
    for i, row in df.iterrows():
        logits = parse_logits(row["Logits"])
        emissions = torch.tensor(logits)
        decoded = decoder.decode(emissions)
        lm_predictions.append(decoded)
        if (i + 1) % 50 == 0:
            print(f"  Decoded {i + 1}/{len(df)} sentences")

    df["LM Predictions"] = lm_predictions
    df["CER_LM"] = df.apply(
        lambda r: (
            Levenshtein.distance(r["True Sentences"], r["LM Predictions"])
            / max(len(r["True Sentences"]), 1)
        ),
        axis=1,
    )
    df["WER_LM"] = df.apply(
        lambda r: (
            Levenshtein.distance(
                r["True Sentences"].split(),
                r["LM Predictions"].split(),
            )
            / max(len(r["True Sentences"].split()), 1)
        ),
        axis=1,
    )

    header = f"{'Subject':<12} {'CER':>12} {'CER_LM':>10} {'WER_LM':>10}"
    print(f"\n{header}")
    print("-" * len(header))
    for subject in sorted(df["Subject"].unique()):
        sdf = df[df["Subject"] == subject]
        print(
            f"{subject:<12} "
            f"{sdf['CER'].mean():>12.3f} "
            f"{sdf['CER_LM'].mean():>10.3f} "
            f"{sdf['WER_LM'].mean():>10.3f}"
        )

    per_subj = df.groupby("Subject")[["CER", "CER_LM", "WER_LM"]].mean()
    print("-" * len(header))
    print(
        f"{'Overall':<12} "
        f"{per_subj['CER'].mean():>12.3f} "
        f"{per_subj['CER_LM'].mean():>10.3f} "
        f"{per_subj['WER_LM'].mean():>10.3f}"
    )

    output_path = args.output or args.input
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
