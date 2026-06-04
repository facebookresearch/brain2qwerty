#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Extract structured prediction results from LogPreds callback outputs.

The training pipeline saves per-sentence JSON files via the LogPreds callback
(see brain2qwerty/callbacks.py). This script reads those JSON files,
reconstructs predicted and typed sentences from class indices, computes
per-sentence CER and WER for each subject, and exports a single clean CSV
ready for downstream analysis.

Input:
    A callbacks directory containing test_all_sentences.json (or
    val_all_sentences.json), as produced by the LogPreds callback during
    training.

Output:
    A CSV file with columns:
        Subject, Sentence_UID, True Sentences, Typed Sentences,
        Model Predictions, Logits, CER, WER

Usage:
    python extract_predictions.py --input <callbacks_dir> --output predictions.csv
    python extract_predictions.py --input <callbacks_dir>  # writes to <callbacks_dir>/predictions.csv
"""

import argparse
import json
import os

import Levenshtein
import pandas as pd

CHAR_MAP = {
    0: "s",
    1: "o",
    2: "t",
    3: "e",
    4: "n",
    5: "c",
    6: "i",
    7: "a",
    8: " ",
    9: "d",
    10: "l",
    11: "r",
    12: "b",
    13: "@",
    14: "z",
    15: "v",
    16: "f",
    17: "m",
    18: "u",
    19: "h",
    20: "p",
    21: "g",
    22: "q",
    23: "w",
    24: "x",
    25: "y",
    26: "j",
    27: "k",
    28: "9",
}


def reconstruct_sentence(class_indices):
    """Convert a list of predicted class indices back to a string."""
    return "".join(CHAR_MAP.get(idx, "?") for idx in class_indices)


def compute_cer(true, pred):
    if len(true) == 0:
        return float("nan")
    return Levenshtein.distance(true, pred) / len(true)


def compute_wer(true, pred):
    true_words = true.split()
    pred_words = pred.split()
    if len(true_words) == 0:
        return float("nan")
    return Levenshtein.distance(true_words, pred_words) / len(true_words)


def extract_subject(sentence_uid):
    """Extract a subject identifier from the sentence UID.

    UIDs follow the pattern: <trial_id>_Pinet2024Meg_subject-<SX>_session-..._task-...
    """
    parts = sentence_uid.split("_")
    for part in parts:
        if part.startswith("subject-"):
            return part.replace("subject-", "")
    return "unknown"


def process_json(json_path):
    """Load a LogPreds JSON and return a DataFrame with predictions and metrics."""
    with open(json_path) as f:
        data = json.load(f)

    rows = []
    for sentence_uid, entry in data.items():
        true_sentence = entry["true"]
        typed_sentence = reconstruct_sentence(entry["typed"])
        model_prediction = reconstruct_sentence(entry["pred"])
        subject = extract_subject(sentence_uid)

        rows.append(
            {
                "Subject": subject,
                "Sentence_UID": sentence_uid,
                "True Sentences": true_sentence,
                "Typed Sentences": typed_sentence,
                "Model Predictions": model_prediction,
                "Logits": entry["logits"],
                "CER": compute_cer(true_sentence, model_prediction),
                "WER": compute_wer(true_sentence, model_prediction),
            }
        )

    df = (
        pd.DataFrame(rows).sort_values(["Subject", "Sentence_UID"]).reset_index(drop=True)
    )
    return df


def print_summary(df):
    """Print per-subject and overall CER/WER summary."""
    print(f"\n{'Subject':<12} {'N sentences':>12} {'Mean CER':>10} {'Mean WER':>10}")
    print("-" * 46)
    for subject in sorted(df["Subject"].unique()):
        sdf = df[df["Subject"] == subject]
        print(
            f"{subject:<12} {len(sdf):>12} {sdf['CER'].mean():>10.3f} {sdf['WER'].mean():>10.3f}"
        )

    print("-" * 46)
    per_subj_cer = df.groupby("Subject")["CER"].mean()
    per_subj_wer = df.groupby("Subject")["WER"].mean()
    print(
        f"{'Overall':<12} {len(df):>12} {per_subj_cer.mean():>10.3f} {per_subj_wer.mean():>10.3f}"
    )
    print(f"  (n = {df['Subject'].nunique()} subjects)\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract predictions from LogPreds callback JSON"
    )
    parser.add_argument(
        "--input", required=True, help="Path to callbacks directory or JSON file"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: <input>/predictions.csv)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "val"],
        help="Which split to process (default: test)",
    )
    args = parser.parse_args()

    if os.path.isdir(args.input):
        json_path = os.path.join(args.input, f"{args.split}_all_sentences.json")
    else:
        json_path = args.input

    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found")
        return

    print(f"Loading {json_path} ...")
    df = process_json(json_path)
    print_summary(df)

    output_path = args.output or os.path.join(
        os.path.dirname(json_path), "predictions.csv"
    )
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()