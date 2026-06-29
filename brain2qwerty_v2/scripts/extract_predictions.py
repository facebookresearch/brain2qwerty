# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import Levenshtein
import pandas as pd

# The headline metric reported for V2 (V1 reports CER instead).
HEADLINE_METRIC = "WER"


def compute_cer(true: str, pred: str) -> float:
    true, pred = str(true), str(pred)
    if len(true) == 0:
        return float("nan")
    return Levenshtein.distance(true, pred) / len(true)


def compute_wer(true: str, pred: str) -> float:
    true_words, pred_words = str(true).split(), str(pred).split()
    if len(true_words) == 0:
        return float("nan")
    return Levenshtein.distance(true_words, pred_words) / len(true_words)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    # Recompute CER/WER from the saved true/pred text so the script is
    # self-contained (CTC_CER and SemER are read from the CSV when present).
    df = df.copy()
    df["CER"] = df.apply(lambda r: compute_cer(r["true_text"], r["pred_text"]), axis=1)
    df["WER"] = df.apply(lambda r: compute_wer(r["true_text"], r["pred_text"]), axis=1)
    if "ctc_text" in df.columns:
        df["CTC_CER"] = df.apply(
            lambda r: compute_cer(r["true_text"], r["ctc_text"]), axis=1
        )
    return df


def print_summary(df: pd.DataFrame) -> None:
    cols = ["CER", "WER"] + [c for c in ("CTC_CER", "SemER") if c in df.columns]
    has_subject = "subject" in df.columns and df["subject"].notna().any()
    group = df["subject"] if has_subject else pd.Series("all", index=df.index)

    header = f"{'Subject':<22} {'N':>6} " + " ".join(f"{c:>9}" for c in cols)
    print(f"\n{header}\n" + "-" * len(header))
    for subject in sorted(group.unique()):
        sdf = df[group == subject]
        vals = " ".join(f"{sdf[c].mean():>9.3f}" for c in cols)
        print(f"{str(subject):<22} {len(sdf):>6} {vals}")
    print("-" * len(header))

    # Two aggregates: sentence-wise (mean over all sentences) and the reported
    # per-subject average (mean per subject, then mean across subjects).
    sent = " ".join(f"{df[c].mean():>9.3f}" for c in cols)
    print(f"{'Overall (sentence)':<22} {len(df):>6} {sent}")
    if has_subject:
        per_subj = df.groupby("subject")[cols].mean()
        macro = " ".join(f"{per_subj[c].mean():>9.3f}" for c in cols)
        n_subj = df["subject"].nunique()
        print(f"{'Overall (per-subject)':<22} {n_subj:>6} {macro}")
        # Headline metric (V2 = WER) with standard error of the mean across subjects.
        m = per_subj[HEADLINE_METRIC]
        sem = m.std(ddof=1) / (len(m) ** 0.5)
        print(
            f"\n==> {HEADLINE_METRIC} = {m.mean():.1%} +/- {sem:.1%} (SEM) "
            f"across {n_subj} subjects"
        )
    print()


def main(argv: list[str] | None = None) -> None:
    # Purpose of this file: read the per-sentence predictions CSV written by
    # PredictionCSVCallback (true/ctc/pred text per sentence) and report the
    # final metrics, both sentence-wise and averaged per subject (the headline
    # WER). Mirrors brain2qwerty_v1/scripts/extract_predictions.py.
    parser = argparse.ArgumentParser(description="Summarize V2 predictions CSV")
    parser.add_argument("--input", required=True, help="predictions CSV or its directory")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--output", default=None, help="optional per-subject summary CSV")
    args = parser.parse_args(argv)

    csv_path = (
        os.path.join(args.input, f"predictions_{args.split}.csv")
        if os.path.isdir(args.input)
        else args.input
    )
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    print(f"Reading {args.split} predictions from {csv_path}")
    df = summarize(pd.read_csv(csv_path))
    n_subj = df["subject"].nunique() if "subject" in df.columns else 1
    print(
        f"Scoring {len(df)} sentences across {n_subj} subjects "
        f"(headline metric: {HEADLINE_METRIC})"
    )
    print_summary(df)
    if args.output and "subject" in df.columns:
        cols = ["CER", "WER"] + [c for c in ("CTC_CER", "SemER") if c in df.columns]
        df.groupby("subject")[cols].mean().to_csv(args.output)
        print(f"Saved per-subject summary to {args.output}")


if __name__ == "__main__":
    main()
