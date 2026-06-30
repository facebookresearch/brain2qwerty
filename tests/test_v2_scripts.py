# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import pandas as pd
from brain2qwerty_v2.scripts.extract_predictions import (
    compute_cer,
    compute_wer,
    main,
    print_summary,
    summarize,
)

def test_compute_cer_and_wer():
    assert compute_cer("brain", "brain") == 0.0
    assert compute_cer("brain", "brains") == 0.2  # 1 insertion / 5 reference chars
    assert math.isnan(compute_cer("", "anything"))  
    assert compute_wer("the cat sat", "the cat sat") == 0.0
    assert compute_wer("the cat sat", "the dog sat") == 1 / 3  # 1 of 3 words wrong
    assert math.isnan(compute_wer("", "the cat"))


def test_summarize_adds_error_columns():
    """summarize() recomputes CER/WER (and CTC_CER when available) from raw text.
    this is what extract_predictions reports, recomputed from the saved true/pred(/ctc)
    text rather than trusted from upstream, so a stale or wrong metric column can't 
    silently leak into the headline numbers.
    """
    df = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S2"],
            "true_text": ["hello world", "brain", "qwerty"],
            "pred_text": ["hello world", "brains", "qwerty"],
            "ctc_text": ["hello word", "brain", "qwerty"],
        }
    )
    out = summarize(df)

    assert list(out["CER"]) == [0.0, compute_cer("brain", "brains"), 0.0]
    assert list(out["WER"]) == [0.0, compute_wer("brain", "brains"), 0.0]
    assert "CTC_CER" in out.columns
    assert out.loc[0, "CTC_CER"] == compute_cer("hello world", "hello word")


def test_summarize_without_ctc_column():
    df = pd.DataFrame({"true_text": ["ab"], "pred_text": ["ab"]})
    out = summarize(df)
    assert "CTC_CER" not in out.columns


def test_print_summary_runs_and_reports_headline_wer(capsys):
    """print_summary prints per-subject rows plus the headline WER +/- SEM.

    Why it matters: this is the human-readable report researchers read off
    directly, so it should at least run cleanly on a realistic multi-subject
    frame and surface the headline metric (WER for V2, per AGENT_README).
    """
    df = summarize(
        pd.DataFrame(
            {
                "subject": ["S1", "S1", "S2"],
                "true_text": ["a b", "c d", "e f"],
                "pred_text": ["a b", "c x", "e g"],
            }
        )
    )
    print_summary(df)
    out = capsys.readouterr().out
    assert "WER" in out
    assert "S1" in out and "S2" in out
    assert "Overall (per-subject)" in out


def test_main_reads_predictions_csv_from_directory(tmp_path, capsys):
    """End-to-end check of the CLI entry point's path resolution and CSV output.

    Why it matters: main() is invoked as `--input <results>/callbacks`, a
    directory, and must find `predictions_<split>.csv` inside it -- this is the
    contract the training pipeline's output layout has to satisfy. It also
    checks the optional `--output` per-subject summary is written.
    """
    results_dir = tmp_path / "callbacks"
    results_dir.mkdir()
    pd.DataFrame(
        {
            "subject": ["S1", "S2"],
            "true_text": ["hello world", "brain qwerty"],
            "pred_text": ["hello world", "brain qwerty"],
        }
    ).to_csv(results_dir / "predictions_val.csv", index=False)

    out_csv = tmp_path / "summary.csv"
    main(["--input", str(results_dir), "--split", "val", "--output", str(out_csv)])

    printed = capsys.readouterr().out
    assert "predictions_val.csv" in printed
    assert out_csv.exists()
    summary = pd.read_csv(out_csv, index_col=0)
    assert set(summary.index) == {"S1", "S2"}
    assert (summary["WER"] == 0.0).all()