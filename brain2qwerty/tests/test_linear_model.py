# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd

from brain2qwerty.linear_model import CONTROL, EXCLUDE, MERGE, apply_subject_merge


def test_apply_subject_merge():
    subjects = [
        "Pinet2024Meg/S1",
        "Pinet2024Meg/S18",
        "Pinet2024Meg/S23",
        "Pinet2024Meg/S14",
        "Pinet2024Meg/S2",
        "Pinet2024Meg/S11122024",
    ]
    events = pd.DataFrame({"subject": subjects, "value": range(len(subjects))})
    result = apply_subject_merge(events)

    remaining = set(result["subject"].unique())
    assert "Pinet2024Meg/S23" not in remaining
    assert "Pinet2024Meg/S11122024" not in remaining
    assert "Pinet2024Meg/S1" in remaining
    assert "Pinet2024Meg/S18" not in remaining


def test_merge_mapping_is_consistent():
    for src, dst in MERGE.items():
        assert src.startswith("Pinet2024Meg/S")
        assert dst.startswith("Pinet2024Meg/S")


def test_exclude_and_control_are_disjoint():
    assert EXCLUDE.isdisjoint(CONTROL)
