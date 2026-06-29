# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp

import numpy as np
import pandas as pd
from exca import MapInfra
from tqdm import tqdm

from neuralset.events.study import EventsTransform
from neuralset.events.transforms.utils import DeterministicSplitter
from neuralset.extractors.text import BaseText

from .utils import key_to_int

logger = logging.getLogger(__name__)


class EnglishBCBLPreprocessing(EventsTransform):
    """Clean the raw EnglishBCBL events and build the integer CTC target.

    Drops practice trials, normalises buttons (``<space>`` -> ``&``), removes
    unmapped/perception events, and stores the per-sentence ``typed_label`` (the
    space-separated key ids used as the CTC target) plus a stable ``button_UID``.
    """

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        # drop the four practice trials of each session
        events = events[~events.trial_id.isin([0, 1, 2, 3])]
        events["sentence_UID"] = (
            events["trial_id"].astype(str) + "_" + events["timeline"].astype(str)
        )

        # normalise buttons: space -> "&", drop special/number tokens
        events = events[~events.button.isin(["<special>", "<number>"])]
        events.loc[events.button == "<space>", "button"] = "&"

        # drop keystrokes whose button is outside the CTC vocabulary
        unmapped = (events.type == "Keystroke") & ~events.button.isin(key_to_int)
        if unmapped.any():
            logger.info("Dropping %d keystroke(s) with unmapped buttons", unmapped.sum())
            events = events[~unmapped]

        # integer key id per keystroke
        button_events = events[events.type == "Keystroke"]
        events["typed_key_int"] = -1
        events["typed_key_int"] = events["typed_key_int"].astype(int)
        events.loc[button_events.index, "typed_key_int"] = button_events.button.map(
            key_to_int
        )

        # build the space-separated CTC target per sentence (skip near-empty ones)
        uids_to_drop: list[str] = []
        label_by_uid: dict[str, str] = {}
        for uid, group in tqdm(events.groupby("sentence_UID"), desc="Typed labels"):
            if "nan" in uid:
                continue
            buttons = group[group.type == "Keystroke"]
            if len(buttons) == 0 or len(buttons) < 0.5 * len(group):
                uids_to_drop.append(uid)
                continue
            typed_seq_ids = [int(i) for i in buttons.typed_key_int.values]
            assert sum(i == -1 for i in typed_seq_ids) == 0, f"Unmapped keys in {uid}"
            label_by_uid[uid] = " ".join(str(i) for i in typed_seq_ids)
        events["typed_label"] = events["sentence_UID"].map(label_by_uid)
        if uids_to_drop:
            logger.info(
                "Dropping %d sentences with too few keystrokes", len(uids_to_drop)
            )
            events = events[~events.sentence_UID.isin(uids_to_drop)]

        # keep MEG + keystrokes + sentences that carry a label; drop perception rows
        events = events[~(events.is_percep.eq(True) & (events.type != "Meg"))]
        events = events[events.type.isin(["Sentence", "Keystroke", "Meg"])]
        is_sentence = events["type"] == "Sentence"
        has_label = events["typed_label"].notna() & (events["typed_label"] != "")
        events = events[~is_sentence | has_label]

        # stable per-keystroke id (ordered within each sentence)
        keystroke_mask = events["type"] == "Keystroke"
        ks = events.loc[keystroke_mask].sort_values("start")
        if len(ks) > 0:
            counter = ks.groupby("sentence_UID").cumcount() + 1
            events.loc[keystroke_mask, "button_UID"] = (
                ks["sentence_UID"] + "_button_" + counter.astype(str)
            )
        return events


class Brain2QwertyV2Splitter(EventsTransform):
    """Train/val/test split by unique sentence text (no text leakage).

    Each unique Sentence ``text`` is hashed to a split via ``DeterministicSplitter``
    and the assignment is propagated to every row sharing the same ``sentence_UID``.
    """

    deterministic_splitter: DeterministicSplitter

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        sents = events[events["type"] == "Sentence"].copy()
        text_to_split = {
            text: self.deterministic_splitter(str(text))
            for text in sents["text"].dropna().unique()
        }
        uid_to_split = {
            row["sentence_UID"]: text_to_split[row["text"]]
            for _, row in sents.iterrows()
            if pd.notna(row["text"]) and row["text"] in text_to_split
        }
        events["split"] = events["sentence_UID"].map(uid_to_split)
        return events


class WordCreator(EventsTransform):
    """Create one Word event per whitespace token of each Sentence.

    Each Word inherits the parent Sentence's identifiers/timing and records its
    ``word_order`` and left ``context`` so contextualised text embeddings can be
    computed as the contrastive target.
    """

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        sents = events[events["type"] == "Sentence"]
        if sents.empty:
            return events
        inherit_cols = [
            c
            for c in ("timeline", "subject", "start", "duration", "sentence_UID")
            if c in events.columns
        ]
        word_rows: list[dict] = []
        for _, row in sents.iterrows():
            words = str(row["text"]).strip().split()
            for idx, word in enumerate(words):
                wr = {
                    "type": "Word",
                    "text": word,
                    "sentence": str(row["text"]).strip(),
                    "context": " ".join(words[: idx + 1]),
                    "word_order": idx,
                }
                wr.update({col: row[col] for col in inherit_cols})
                word_rows.append(wr)
        if not word_rows:
            return events
        return pd.concat([events, pd.DataFrame(word_rows)], ignore_index=True)


class SentenceKeySeq(BaseText):
    """Turn each sentence into the integer character sequence the CTC head predicts.

    Two ways to build that sequence:
    - ``mode="typed_label"`` uses what the participant actually typed (the integer
      sequence precomputed per sentence in ``event.extra["typed_label"]``).
    - ``mode="sentence_text"`` uses the reference sentence text: lowercase it, map
      spaces to ``&`` and each character to its index via ``key_to_int``.
    """

    event_types: str | tuple[str, ...] = "Sentence"
    mode: tp.Literal["typed_label", "sentence_text"] = "typed_label"

    infra: MapInfra = MapInfra(version="v5")

    @infra.apply(
        item_uid=lambda event: str(event.text),
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
        cache_type="MemmapArrayFile",
    )
    def _get_data(self, events: list[tp.Any]) -> tp.Iterator[np.ndarray]:
        if len(events) > 1:
            events = tqdm(events, desc="Sequence labels")  # type: ignore
        for event in events:
            yield self.get_embedding(event)

    def get_embedding(self, event) -> np.ndarray:
        if self.mode == "typed_label":
            return np.array(
                [int(i) for i in event.extra["typed_label"].split(" ")], dtype=np.int32
            )
        text = str(event.text).lower().replace(" ", "&")
        seq = [key_to_int[ch] for ch in text if ch in key_to_int]
        if not seq:
            raise ValueError(f"Empty target for text={event.text!r}")
        return np.array(seq, dtype=np.int32)
