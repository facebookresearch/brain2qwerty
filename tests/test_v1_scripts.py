# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json

import pandas as pd

from brain2qwerty_v1.scripts.extract_predictions import (
    compute_cer,
    compute_wer,
    process_json,
    reconstruct_sentence,
)
from brain2qwerty_v1.utils import select_participants


def test_reconstruct_and_error_rates():
    # Building blocks of the CSV export: class indices -> characters via CHAR_INDEX,
    # and the CER/WER definitions (normalised Levenshtein) used in the report.
    assert reconstruct_sentence([0, 1, 2]) == "sot"  # CHAR_INDEX: 0=s,1=o,2=t
    assert compute_cer("sote", "sote") == 0.0
    assert compute_cer("sote", "soto") == 0.25  # 1 of 4 chars wrong
    assert compute_wer("hola mundo", "hola mundo") == 0.0


def test_process_json(tmp_path):
    """End-to-end check of the callback-JSON -> CSV conversion.

    Why it matters: this is the contract scripts/extract_predictions exposes. It
    confirms the expected columns/order, that the subject is parsed out of the
    UID, that class indices are decoded back to text, and that CER is computed —
    i.e. the artifact downstream analysis (and ngram_decoding) consumes is correct.
    """
    data = {
        "65.0_subject-S1_block1": {
            "true": "sot",
            "pred": [0, 1, 2],
            "typed": [0, 1, 2],
            "logits": [[0.0] * 29] * 3,
        }
    }
    p = tmp_path / "test_all_sentences.json"
    p.write_text(json.dumps(data))
    df = process_json(str(p))
    assert list(df.columns)[:5] == [
        "Subject",
        "Sentence_UID",
        "True Sentences",
        "Typed Sentences",
        "Model Predictions",
    ]
    assert df.loc[0, "Subject"] == "S1"
    assert df.loc[0, "Model Predictions"] == "sot"
    assert df.loc[0, "CER"] == 0.0


def test_select_participants():
    # The 19-participant cohort is defined here: recordings of the same person are
    # merged (S18 -> S1), the metallic-implant subject (S23) and the no-keyboard
    # control sessions (date-named, e.g. S11122024) are dropped. This guards that
    # logic so per-subject metrics are computed over the right participants.
    df = pd.DataFrame(
        {
            "subject": [
                "Pinet2024Meg/S18",  # merged -> S1
                "Pinet2024Meg/S23",  # excluded
                "Pinet2024Meg/S11122024",  # control -> removed
                "Pinet2024Meg/S2",  # kept
            ],
            "type": ["Keystroke"] * 4,
        }
    )
    out = select_participants(df)
    subjects = set(out["subject"])
    assert "Pinet2024Meg/S1" in subjects
    assert "Pinet2024Meg/S2" in subjects
    assert "Pinet2024Meg/S23" not in subjects
    assert "Pinet2024Meg/S11122024" not in subjects
