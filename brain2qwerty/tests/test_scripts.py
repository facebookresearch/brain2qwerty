# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json

import pandas as pd

from brain2qwerty.scripts.extract_predictions import (
    compute_cer,
    compute_wer,
    extract_subject,
    process_json,
    reconstruct_sentence,
)


def test_reconstruct_sentence():
    assert reconstruct_sentence([0, 1, 2, 3]) == "sote"
    assert reconstruct_sentence([8]) == " "
    assert reconstruct_sentence([]) == ""


def test_compute_cer():
    assert compute_cer("abc", "abc") == 0.0
    assert compute_cer("abc", "axc") > 0.0
    assert compute_cer("", "x") != compute_cer("", "x")  # nan


def test_compute_wer():
    assert compute_wer("the cat", "the cat") == 0.0
    assert compute_wer("the cat", "the dog") == 0.5


def test_extract_subject():
    uid = "3.0_Pinet2024Meg_subject-S1_session-1_task-block1"
    assert extract_subject(uid) == "S1"
    assert extract_subject("no_subject_here") == "unknown"


def test_process_json(tmp_path):
    data = {
        "3.0_Pinet2024Meg_subject-S1_session-1_task-block1": {
            "pred": [0, 1, 2],
            "typed": [0, 1, 3],
            "true": "sot",
            "logits": [[0.0] * 29] * 3,
        },
        "4.0_Pinet2024Meg_subject-S2_session-1_task-block1": {
            "pred": [7, 8, 12],
            "typed": [7, 8, 12],
            "true": "a b",
            "logits": [[0.0] * 29] * 3,
        },
    }
    json_path = tmp_path / "test_all_sentences.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    df = process_json(str(json_path))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert set(df.columns) >= {"Subject", "Sentence_UID", "True Sentences", "CER", "WER"}
    assert df["Subject"].tolist() == ["S1", "S2"]
