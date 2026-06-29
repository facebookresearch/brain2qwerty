# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from neuralset.events.study import EventsTransform

from .utils import select_participants


class SpanishBCBLPreprocessing(EventsTransform):
    """Clean the raw SpanishBCBL events and build sentence-level metadata."""

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        events["type"] = events["type"].replace(
            {"Button": "Keystroke", "DetectedButton": "Keystroke"}
        )

        # drop the two practice trials of each block
        events = events[~events["trial_id"].isin([0.0, 1.0])]

        # keep the 19 unique participants (drop controls/excluded, merge duplicates)
        events = select_participants(events)
        events["subject"] = pd.factorize(events["subject"])[0]

        # unique id per sentence and per keystroke
        if "sentence_UID" not in events.columns:
            events["sentence_UID"] = (
                events["trial_id"].astype(str) + "_" + events["timeline"]
            )
        events = events[
            events["sentence_UID"] != "65.0_Pinet2024Meg_subject-S1_session-1_task-block1"
        ]
        events["button_count"] = events.groupby("sentence_UID")["type"].transform(
            lambda x: (x == "Keystroke").cumsum()
        )
        events["button_unique_id"] = (
            events["sentence_UID"] + "_" + events["button_count"].astype(str)
        )
        events = events.drop(columns="button_count")
        events.loc[events["type"] != "Keystroke", "button_unique_id"] = np.nan
        events.loc[events["type"] != "Keystroke", "sentence"] = np.nan

        buttons = events[events["type"] == "Keystroke"]
        assert len(buttons) == len(buttons["button_unique_id"].unique())

        # rebuild one Sentence event per keystroke group (spanning its keystrokes)
        sentences_uids = buttons["sentence_UID"].unique()
        events = events[events["type"] != "Sentence"]
        grouped_events = events.groupby("sentence_UID")
        updated_sentences = []
        for suid in sentences_uids:
            df = grouped_events.get_group(suid)
            button_events = df[df["type"] == "Keystroke"]
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
            events = pd.concat([events, pd.DataFrame(updated_sentences)])
            events.reset_index(drop=True, inplace=True)

        sent_mask = events["type"] == "Sentence"
        if "is_image" in events.columns:
            sent_mask = sent_mask & (events["is_image"] == False)  # noqa: E712
        sentences = events[sent_mask]
        assert len(sentences) == len(sentences["sentence_UID"].unique())

        # propagate the ground-truth sentence text to the Sentence events
        button_df = events[events["type"] == "Keystroke"]
        sentence_dict = button_df.set_index("sentence_UID")["sentence"].to_dict()
        events["text"] = events.apply(
            lambda row: (
                sentence_dict.get(row["sentence_UID"], None)
                if row["type"] == "Sentence"
                else row["text"]
            ),
            axis=1,
        )

        # typed string per sentence (special tokens mapped to single characters)
        sentence_typed = (
            events[events["type"] == "Keystroke"]
            .groupby("sentence_UID")["button"]
            .apply("".join)
            .reset_index(name="sentence_typed")
        )
        events = events.merge(sentence_typed, on="sentence_UID", how="left")
        events["sentence_typed"] = events.apply(
            lambda row: (
                row["sentence_typed"]
                if row["type"] in ["Keystroke", "Sentence"]
                else np.nan
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

        # stable per-keystroke id (ordered within each sentence)
        keystroke_mask = events["type"] == "Keystroke"
        ks = events.loc[keystroke_mask].sort_values("start")
        if len(ks) > 0:
            counter = ks.groupby("sentence_UID").cumcount() + 1
            events.loc[keystroke_mask, "button_UID"] = (
                ks["sentence_UID"] + "_button_" + counter.astype(str)
            )
        return events


class Brain2QwertyV1Splitter(EventsTransform):
    """Sentence-level train/val/test split with no paraphrase leakage.

    Sentences are clustered by TF-IDF cosine similarity, then clusters are
    allocated greedily to the splits until the target ratios (in keystrokes)
    are met, so similar sentences always land in the same split.
    """

    splitting_ratios: tuple = (0.8, 0.1, 0.1)
    seed: int = 1
    threshold: float = 0.5

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        buttons = events[events["type"] == "Keystroke"]
        unique_sentences = buttons["sentence"].unique()
        random.seed(self.seed)

        # cluster paraphrase-similar sentences via TF-IDF cosine similarity
        tfidf = TfidfVectorizer().fit_transform(unique_sentences)
        sim = cosine_similarity(tfidf)
        clusters: list[list[int]] = []
        visited: set[int] = set()
        for i in range(sim.shape[0]):
            if i in visited:
                continue
            cluster = {i}
            expanded = True
            while expanded:
                expanded = False
                for idx in list(cluster):
                    for j in range(sim.shape[1]):
                        if j not in cluster and sim[idx, j] > self.threshold:
                            cluster.add(j)
                            expanded = True
            visited.update(cluster)
            clusters.append(list(cluster))
        random.shuffle(clusters)

        # allocate clusters to splits to hit the target keystroke ratios
        total = len(buttons)
        sizes = {
            "train": int(self.splitting_ratios[0] * total),
            "val": int(self.splitting_ratios[1] * total),
            "test": total
            - int(self.splitting_ratios[0] * total)
            - int(self.splitting_ratios[1] * total),
        }
        current = {"train": 0, "val": 0, "test": 0}
        sentence_to_split: dict[str, str] = {}
        for cluster in clusters:
            cluster_sents = [unique_sentences[idx] for idx in cluster]
            cluster_size = len(buttons[buttons["sentence"].isin(cluster_sents)])
            assigned = "test"
            for split in ["train", "val", "test"]:
                if current[split] + cluster_size <= sizes[split]:
                    current[split] += cluster_size
                    assigned = split
                    break
            for s in cluster_sents:
                sentence_to_split[s] = assigned

        events["split"] = events["sentence"].map(sentence_to_split)
        return events
