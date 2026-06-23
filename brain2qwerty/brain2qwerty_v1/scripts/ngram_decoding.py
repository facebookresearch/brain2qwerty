# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import ast

import kenlm
import Levenshtein
import pandas as pd
import torch

from ..utils import NGRAM_CHAR_INDEX


class BeamState:
    def __init__(
        self, sentence: str, score: float, lm_state: "kenlm.State | None" = None
    ):
        self.sentence = sentence
        self.score = score
        self.lm_state = lm_state or kenlm.State()


class BeamDecoder:
    """Character-level beam search decoder with a KenLM model."""

    def __init__(
        self,
        lm,
        beam_size: int = 30,
        max_labels_per_timestep: int = 5,
        lm_weight: float = 5.0,
    ):
        self.lm = lm
        self.beam_size = beam_size
        self.max_labels_per_timestep = max_labels_per_timestep
        self.lm_weight = lm_weight

    def decode(self, emissions: torch.Tensor) -> str:
        beam = [BeamState(sentence="", score=0.0)]
        self.lm.BeginSentenceWrite(beam[0].lm_state)

        for logits in emissions:
            probs = torch.softmax(logits, dim=0)
            top_idx = probs.argsort(descending=True)[: self.max_labels_per_timestep]
            new_beam = []
            for hyp in beam:
                for idx in top_idx:
                    char = NGRAM_CHAR_INDEX[idx.item()]
                    if char.isdigit():
                        continue
                    new_state = kenlm.State()
                    lm_score = self.lm.BaseScore(hyp.lm_state, char, new_state)
                    score = hyp.score + self.lm_weight * lm_score + torch.log(probs[idx])
                    new_beam.append(BeamState(hyp.sentence + char, score, new_state))
            beam = sorted(new_beam, key=lambda x: x.score, reverse=True)[: self.beam_size]
        return beam[0].sentence.replace("&", " ")


def parse_logits(logits_str):
    return ast.literal_eval(logits_str) if isinstance(logits_str, str) else logits_str


def main(argv: list[str] | None = None) -> None:
    # What this does: take the predictions CSV produced by extract_predictions
    # (it carries the per-keystroke ``Logits``) and rescore each sentence with a
    # character-level KenLM n-gram via beam search, writing LM predictions and
    # their CER/WER back out. Just point --input at the CSV and --lm at the ARPA.
    parser = argparse.ArgumentParser(description="N-gram LM post-processing")
    parser.add_argument("--input", required=True, help="input predictions CSV")
    parser.add_argument("--lm", required=True, help="path to a KenLM .arpa model")
    parser.add_argument("--output", default=None, help="output CSV path")
    parser.add_argument("--lm-weight", type=float, default=5.0)
    parser.add_argument("--beam-size", type=int, default=30)
    parser.add_argument("--max-labels", type=int, default=5)
    args = parser.parse_args(argv)

    decoder = BeamDecoder(
        kenlm.Model(args.lm),
        beam_size=args.beam_size,
        max_labels_per_timestep=args.max_labels,
        lm_weight=args.lm_weight,
    )
    df = pd.read_csv(args.input)

    df["LM Predictions"] = [
        decoder.decode(torch.tensor(parse_logits(row["Logits"])))
        for _, row in df.iterrows()
    ]
    df["CER_LM"] = df.apply(
        lambda r: Levenshtein.distance(r["True Sentences"], r["LM Predictions"])
        / max(len(r["True Sentences"]), 1),
        axis=1,
    )
    df["WER_LM"] = df.apply(
        lambda r: Levenshtein.distance(
            r["True Sentences"].split(), r["LM Predictions"].split()
        )
        / max(len(r["True Sentences"].split()), 1),
        axis=1,
    )

    per_subj = df.groupby("Subject")[["CER", "CER_LM", "WER_LM"]].mean()
    print(per_subj.to_string())
    print(
        f"\nOverall  CER={per_subj['CER'].mean():.3f}  "
        f"CER_LM={per_subj['CER_LM'].mean():.3f}  WER_LM={per_subj['WER_LM'].mean():.3f}"
    )

    output_path = args.output or args.input
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
