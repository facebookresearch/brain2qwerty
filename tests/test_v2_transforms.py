# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
import pandas as pd
import pytest
from brain2qwerty_v2.transforms import (
    Brain2QwertyV2Splitter,
    EnglishBCBLPreprocessing,
    SentenceKeySeq,
    WordCreator,
)
from brain2qwerty_v2.utils import key_to_int
from neuralset.events.transforms.utils import DeterministicSplitter

TIMELINE = "S1_ses1_block1"


def _uid(trial_id: int) -> str:
    return f"{float(trial_id)}_{TIMELINE}"


def _row(type_, trial_id, start, duration=0.1, button=None, text=None, is_percep=False):
    return dict(
        type=type_,
        trial_id=float(trial_id),
        timeline=TIMELINE,
        subject="S1",
        button=button,
        text=text,
        is_percep=is_percep,
        start=start,
        duration=duration,
    )


def _synthetic_events() -> pd.DataFrame:
    rows: list[dict] = []
    t = 0.0

    def add_keystrokes(trial_id, buttons, is_percep=False):
        nonlocal t
        for b in buttons:
            rows.append(_row("Keystroke", trial_id, t, button=b, is_percep=is_percep))
            t += 0.15

    # the four practice trials must be dropped wholesale, regardless of content
    for trial_id in (0, 1, 2, 3):
        rows.append(_row("Sentence", trial_id, t, text="practice"))
        add_keystrokes(trial_id, ["p", "r"])

    # trial 4: ordinary sentence "ab cd" (space normalises to "&"); also carries
    # a Meg event and an unrelated event type that should both be handled
    rows.append(_row("Sentence", 4, t, text="ab cd"))
    rows.append(_row("Meg", 4, t, duration=2.0))
    rows.append(_row("Fixation", 4, t))
    add_keystrokes(4, ["a", "b", "<space>", "c", "d"])

    # trial 5: a key outside the CTC vocabulary ("1") interleaved -> dropped
    rows.append(_row("Sentence", 5, t, text="ace"))
    add_keystrokes(5, ["a", "1", "c", "e"])

    # trial 6: the "<number>"/"<special>" sentinels are dropped outright
    rows.append(_row("Sentence", 6, t, text="ab"))
    add_keystrokes(6, ["a", "<number>", "<special>", "b"])

    # trial 7: a perception (playback, not typed) trial -> dropped wholesale
    rows.append(_row("Sentence", 7, t, text="perceived", is_percep=True))
    add_keystrokes(7, ["p", "e"], is_percep=True)

    # trial 8: a sparse sentence -- too few keystrokes relative to its group -> dropped
    rows.append(_row("Sentence", 8, t, text="sparse"))
    rows.append(_row("Meg", 8, t, duration=2.0))
    add_keystrokes(8, ["a"])

    return pd.DataFrame(rows)


def test_english_preprocessing():
    """Check the core invariants of the EnglishBCBL preprocessing transform.

    Why it matters: this transform builds the integer CTC target every
    downstream component relies on. The asserts pin the contract: the four
    practice trials are dropped, "<space>" normalises to "&", both the
    "<number>"/"<special>" sentinels and any key outside the CTC vocabulary are
    filtered out of the typed_label (not just silently kept), perception
    (playback) trials and sentences with too few keystrokes relative to their
    event group are dropped wholesale, only Sentence/Keystroke/Meg rows survive,
    and button_UID is assigned in temporal order per sentence.
    """
    out = EnglishBCBLPreprocessing().run(_synthetic_events())

    assert not out["trial_id"].isin([0.0, 1.0, 2.0, 3.0]).any()  # practice trials gone

    ks = out[out["type"] == "Keystroke"]
    assert "<space>" not in ks["button"].values
    assert set(ks["button"].unique()) <= set(key_to_int)  # sentinels/unmapped keys gone

    sents = out[out["type"] == "Sentence"].set_index("sentence_UID")

    def _label(text: str) -> str:
        return " ".join(str(key_to_int[c]) for c in text)

    assert sents.loc[_uid(4), "typed_label"] == _label("ab&cd")
    assert sents.loc[_uid(5), "typed_label"] == _label("ace")  # "1" dropped
    assert sents.loc[_uid(6), "typed_label"] == _label("ab")  # sentinels dropped

    # perception trial and the too-sparse trial never make it into the output
    remaining_uids = set(out["sentence_UID"].unique())
    assert _uid(7) not in remaining_uids
    assert _uid(8) not in remaining_uids

    assert set(out["type"].unique()) <= {
        "Sentence",
        "Keystroke",
        "Meg",
    }  # Fixation dropped

    uid4_ks = out[
        (out["sentence_UID"] == _uid(4)) & (out["type"] == "Keystroke")
    ].sort_values("start")
    assert list(uid4_ks["button_UID"]) == [
        f"{_uid(4)}_button_{i}" for i in range(1, len(uid4_ks) + 1)
    ]


def test_v2_splitter_is_deterministic_and_leak_free():
    """The train/val/test splitter assigns one split per sentence, reproducibly.

    Why it matters: every row of a sentence (Keystroke, Meg, Sentence) must land
    in the same split, or the model would leak test sentences into training. A
    non-deterministic split would also make reported numbers irreproducible, so
    we run it twice with the same seed and require identical assignments.
    """
    preprocessed = EnglishBCBLPreprocessing().run(_synthetic_events())
    splitter = Brain2QwertyV2Splitter(
        deterministic_splitter=DeterministicSplitter(
            ratios={"train": 0.7, "val": 0.15, "test": 0.15}, seed=1
        )
    )
    out = splitter.run(preprocessed)

    assert set(out["split"].dropna().unique()) <= {"train", "val", "test"}
    # no leakage: every row of a given sentence_UID has exactly one split
    per_uid_splits = (
        out.dropna(subset=["split"]).groupby("sentence_UID")["split"].nunique()
    )
    assert (per_uid_splits == 1).all()

    out2 = Brain2QwertyV2Splitter(
        deterministic_splitter=DeterministicSplitter(
            ratios={"train": 0.7, "val": 0.15, "test": 0.15}, seed=1
        )
    ).run(EnglishBCBLPreprocessing().run(_synthetic_events()))
    pd.testing.assert_series_equal(
        out.sort_values("button_UID")["split"].reset_index(drop=True),
        out2.sort_values("button_UID")["split"].reset_index(drop=True),
    )


def test_word_creator():
    """WordCreator splits each Sentence into one Word event per token.

    Why it matters: the word-level contrastive branch needs a Word event per
    whitespace token, each carrying its position (``word_order``) and growing
    left context -- the input to the contextualised text embeddings used as the
    contrastive target. This pins that contract plus column inheritance from
    the parent Sentence.
    """
    events = pd.DataFrame(
        [
            dict(
                type="Sentence",
                text="hello there world",
                sentence_UID="uid-1",
                timeline=TIMELINE,
                subject="S1",
                start=0.0,
                duration=1.0,
            )
        ]
    )
    out = WordCreator().run(events)
    words = out[out["type"] == "Word"].sort_values("word_order")

    assert list(words["text"]) == ["hello", "there", "world"]
    assert list(words["word_order"]) == [0, 1, 2]
    assert list(words["context"]) == ["hello", "hello there", "hello there world"]
    assert (words["sentence"] == "hello there world").all()
    # inherited from the parent Sentence row
    assert (words["sentence_UID"] == "uid-1").all()
    assert (words["timeline"] == TIMELINE).all()


def test_sentence_key_seq_get_embedding():
    """SentenceKeySeq turns a sentence into the integer sequence the CTC head predicts.

    Why it matters: this is the only place the two supported target modes are
    defined. "typed_label" must reuse the precomputed per-keystroke sequence
    verbatim; "sentence_text" must lowercase, map spaces to the CTC "&" class,
    silently skip out-of-vocabulary characters, and raise rather than emit an
    empty target.
    """
    typed = SentenceKeySeq(mode="typed_label")
    event = SimpleNamespace(extra={"typed_label": "1 2 3"})
    assert list(typed.get_embedding(event)) == [1, 2, 3]

    text_mode = SentenceKeySeq(mode="sentence_text")
    event = SimpleNamespace(text="Ab Cd")
    expected = [key_to_int[c] for c in "ab&cd"]
    assert list(text_mode.get_embedding(event)) == expected

    with pytest.raises(ValueError):
        text_mode.get_embedding(SimpleNamespace(text="123"))  # no in-vocabulary chars