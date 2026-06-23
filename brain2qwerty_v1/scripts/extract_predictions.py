# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import re

import Levenshtein
import pandas as pd

from ..utils import CHAR_INDEX


def reconstruct_sentence(class_indices) -> str:
    return "".join(CHAR_INDEX.get(idx, "?") for idx in class_indices)


def compute_cer(true: str, pred: str) -> float:
    if len(true) == 0:
        return float("nan")
    return Levenshtein.distance(true, pred) / len(true)


def compute_wer(true: str, pred: str) -> float:
    true_words, pred_words = true.split(), pred.split()
    if len(true_words) == 0:
        return float("nan")
    return Levenshtein.distance(true_words, pred_words) / len(true_words)


# SpanishBCBL recording ids that belong to the same participant are merged during
# preprocessing (see transforms.SpanishBCBLPreprocessing). The sentence_UID keeps
# the original recording label, so apply the same mapping here to score over the
# 19 participants rather than the raw recordings.
SUBJECT_MERGES = {"S18": "S1", "S14": "S4", "S10": "S5", "S21": "S5"}


def extract_subject(sentence_uid: str) -> str:
    # The participant id appears in the UID as an "S<n>" token (e.g. "..._S1_..."
    # or "..._subject-S1_..."); take the first one and fold merged recordings in.
    match = re.search(r"S(\d+)", sentence_uid)
    if match is None:
        return "unknown"
    subject = "S" + match.group(1)
    return SUBJECT_MERGES.get(subject, subject)


def process_json(json_path: str) -> pd.DataFrame:
    with open(json_path) as f:
        data = json.load(f)
    rows = []
    for sentence_uid, entry in data.items():
        true_sentence = entry["true"]
        prediction = reconstruct_sentence(entry["pred"])
        rows.append(
            {
                "Subject": extract_subject(sentence_uid),
                "Sentence_UID": sentence_uid,
                "True Sentences": true_sentence,
                "Typed Sentences": reconstruct_sentence(entry["typed"]),
                "Model Predictions": prediction,
                "Logits": entry["logits"],
                "CER": compute_cer(true_sentence, prediction),
                "WER": compute_wer(true_sentence, prediction),
            }
        )
    return (
        pd.DataFrame(rows).sort_values(["Subject", "Sentence_UID"]).reset_index(drop=True)
    )


def print_summary(df: pd.DataFrame) -> None:
    header = f"{'Subject':<12} {'N':>12} {'CER':>10} {'WER':>10}"
    print(f"\n{header}\n" + "-" * len(header))
    for subject in sorted(df["Subject"].unique()):
        sdf = df[df["Subject"] == subject]
        print(
            f"{subject:<12} {len(sdf):>12} {sdf['CER'].mean():>10.3f} {sdf['WER'].mean():>10.3f}"
        )
    print("-" * len(header))
    per_subj = df.groupby("Subject")[["CER", "WER"]].mean()
    print(
        f"{'Overall':<12} {len(df):>12} {per_subj['CER'].mean():>10.3f} {per_subj['WER'].mean():>10.3f}"
    )
    # Headline metric (V1 = CER) with standard error of the mean across subjects.
    n_subj = len(per_subj)
    cer = per_subj["CER"]
    sem = cer.std(ddof=1) / (n_subj**0.5)
    print(f"  (n = {n_subj} subjects)")
    print(f"\n==> CER = {cer.mean():.1%} +/- {sem:.1%} (SEM) across {n_subj} subjects\n")


def main(argv: list[str] | None = None) -> None:
    # Purpose of this file: turn the raw per-sentence JSON dumped by the
    # LogSentencePredictions callback during val/test into a clean, analysis-ready
    # CSV (one row per sentence, with reconstructed text and CER/WER per subject).
    parser = argparse.ArgumentParser(description="Extract predictions from callback JSON")
    parser.add_argument("--input", required=True, help="callbacks directory or JSON file")
    parser.add_argument("--output", default=None, help="output CSV path")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    args = parser.parse_args(argv)

    json_path = (
        os.path.join(args.input, f"{args.split}_all_sentences.json")
        if os.path.isdir(args.input)
        else args.input
    )
    if not os.path.exists(json_path):
        raise FileNotFoundError(json_path)

    print(f"Reading {args.split} predictions from {json_path}")
    df = process_json(json_path)
    print(
        f"Scoring {len(df)} sentences across {df['Subject'].nunique()} subjects "
        f"(headline metric: CER)"
    )
    print_summary(df)
    output_path = args.output or os.path.join(
        os.path.dirname(json_path), "predictions.csv"
    )
    df.to_csv(output_path, index=False)
    print(f"Saved per-sentence CSV to {output_path}")


if __name__ == "__main__":
    main()
