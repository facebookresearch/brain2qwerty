# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd

from brain2qwerty.splitter import check_leakage, split_events


def _make_events(n=500, n_unique=50):
    return pd.DataFrame(
        {
            "type": ["Button"] * n,
            "sentence": [f"sentence number {i % n_unique}" for i in range(n)],
        }
    )


def test_split_produces_all_splits():
    events = _make_events(n=500, n_unique=50)
    result = split_events(events, ratios=(0.8, 0.1, 0.1), seed=42)
    assert "split" in result.columns
    assert "train" in result["split"].values


def test_no_leakage():
    events = _make_events(n=500, n_unique=50)
    result = split_events(events, ratios=(0.8, 0.1, 0.1), seed=42)
    leaks = check_leakage(result, threshold=0.5)
    assert len(leaks) == 0, f"Found {len(leaks)} leaking pairs"


def test_same_sentence_same_split():
    events = _make_events(n=500, n_unique=50)
    result = split_events(events, ratios=(0.8, 0.1, 0.1), seed=42)
    for sent in result["sentence"].unique():
        mask = result["sentence"] == sent
        splits = result.loc[mask, "split"].unique()
        assert len(splits) == 1


def test_deterministic_with_seed():
    events = _make_events()
    r1 = split_events(events.copy(), ratios=(0.8, 0.1, 0.1), seed=123)
    r2 = split_events(events.copy(), ratios=(0.8, 0.1, 0.1), seed=123)
    assert r1["split"].tolist() == r2["split"].tolist()


def test_different_seeds_differ():
    events = _make_events()
    r1 = split_events(events.copy(), ratios=(0.8, 0.1, 0.1), seed=1)
    r2 = split_events(events.copy(), ratios=(0.8, 0.1, 0.1), seed=2)
    assert r1["split"].tolist() != r2["split"].tolist()
