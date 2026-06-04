# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from collections import defaultdict

import neuralset as ns
import numpy as np
import pandas as pd

from .splitter import check_leakage, split_events  # noqa: F401

BUTTON_MAPPING = {
    "s": 0,
    "o": 1,
    "t": 2,
    "e": 3,
    "n": 4,
    "c": 5,
    "i": 6,
    "a": 7,
    "<space>": 8,
    "d": 9,
    "l": 10,
    "r": 11,
    "b": 12,
    "<special>": 13,
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
    "<number>": 28,
    # Special characters mapped to <special> (class 13)
    "ý": 13,
    "\x14": 13,
    "ü": 13,
    "û": 13,
    "£": 13,
    "¤": 13,
    "-": 13,
    "¿": 13,
    "`": 13,
}

NUM_CLASSES = len(set(BUTTON_MAPPING.values()))

CHAR_INDEX = {
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


def preprocessing(events: pd.DataFrame) -> pd.DataFrame:
    """Clean raw events: remove practice trials, excluded participants,
    merge duplicate subject sessions, and build sentence-level metadata."""
    events = events[~events["trial_id"].isin([0.0, 1.0])]

    control_data = [
        "Pinet2024Meg/S11122024",
        "Pinet2024Meg/S12122024",
        "Pinet2024Meg/S26112024",
        "Pinet2024Meg/S27112024",
        "Pinet2024Meg/S28112024",
    ]
    events = events[~events["subject"].isin(control_data)]
    events = events[events["subject"] != "Pinet2024Meg/S23"]

    events["subject"] = events["subject"].str.replace(
        "Pinet2024Meg/S18", "Pinet2024Meg/S1"
    )
    events["subject"] = events["subject"].str.replace(
        "Pinet2024Meg/S14", "Pinet2024Meg/S4"
    )
    events["subject"] = events["subject"].str.replace(
        "Pinet2024Meg/S10", "Pinet2024Meg/S5"
    )
    events["subject"] = events["subject"].str.replace(
        "Pinet2024Meg/S21", "Pinet2024Meg/S5"
    )

    events["sentence_UID"] = events["trial_id"].astype(str) + "_" + events["timeline"]
    events = events[
        events["sentence_UID"] != "65.0_Pinet2024Meg_subject-S1_session-1_task-block1"
    ]

    events["button_count"] = events.groupby("sentence_UID")["type"].transform(
        lambda x: (x == "Button").cumsum()
    )
    events["button_unique_id"] = (
        events["sentence_UID"] + "_" + events["button_count"].astype(str)
    )
    events = events.drop(columns="button_count")
    events.loc[events["type"] != "Button", "button_unique_id"] = np.nan
    events.loc[events["type"] != "Button", "sentence"] = np.nan

    buttons = events[events["type"] == "Button"]
    assert len(buttons) == len(buttons["button_unique_id"].unique())

    sentences_uids = buttons["sentence_UID"].unique()
    events = events[events["type"] != "Sentence"]
    grouped_events = events.groupby("sentence_UID")
    updated_sentences = []
    for suid in sentences_uids:
        df = grouped_events.get_group(suid)
        button_events = df[df["type"] == "Button"]
        if not button_events.empty:
            sentence = button_events.iloc[0].copy()
            sentence["type"] = "Sentence"
            sentence["duration"] = (
                button_events["stop"].max() - button_events["start"].min()
            )
            sentence["start"] = button_events["start"].min()
            sentence["stop"] = button_events["stop"].max()
            updated_sentences.append(sentence)

    if updated_sentences:
        updated_sentences_df = pd.DataFrame(updated_sentences)
        events = pd.concat([events, updated_sentences_df])
        events.reset_index(drop=True, inplace=True)

    sentences = events[(events["type"] == "Sentence") & (~events["is_image"])]
    assert len(sentences) == len(sentences["sentence_UID"].unique())

    button_df = events[events["type"] == "Button"]
    sentence_dict = button_df.set_index("sentence_UID")["sentence"].to_dict()
    events["text"] = events.apply(
        lambda row: (
            sentence_dict.get(row["sentence_UID"], None)
            if row["type"] == "Sentence"
            else row["text"]
        ),
        axis=1,
    )

    sentence_typed = (
        events[events["type"] == "Button"]
        .groupby("sentence_UID")["button"]
        .apply("".join)
        .reset_index(name="sentence_typed")
    )
    events = events.merge(sentence_typed, on="sentence_UID", how="left")
    events["sentence_typed"] = events.apply(
        lambda row: (
            row["sentence_typed"] if row["type"] in ["Button", "Sentence"] else np.nan
        ),
        axis=1,
    )
    events["sentence_typed"] = (
        events["sentence_typed"]
        .str.replace("<special>", "@", regex=False)
        .str.replace("<space>", " ", regex=False)
        .str.replace("<number>", "9", regex=False)
    )

    columns_to_drop = [
        "word_id",
        "word_index",
        "trigger",
        "time",
        "pressed",
        "key",
        "is_key",
        "stim",
        "char_id",
        "is_left_key",
        "dropped_char_per",
        "context",
    ]
    events = events.drop(columns=[c for c in columns_to_drop if c in events.columns])

    return events


def shuffle_sentences(
    segments: tp.List[ns.segments.Segment],
) -> tp.List[ns.segments.Segment]:
    """Shuffle segments by blocks of sentences (preserving within-sentence order)."""
    segment_dict = defaultdict(list)
    for segment in segments:
        key = (
            segment._trigger["timeline"],
            segment._trigger["sequence_id"],
        )
        segment_dict[key].append(segment)
    keys = list(segment_dict.keys())
    np.random.shuffle(keys)
    return [segment for key in keys for segment in segment_dict[key]]


class ShuffledSegmentDataset(ns.SegmentDataset):
    def shuffle(self):
        self.segments = shuffle_sentences(self.segments)
