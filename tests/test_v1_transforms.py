# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd

from brain2qwerty_v1.transforms import Brain2QwertyV1Splitter, SpanishBCBLPreprocessing

_SENTENCES = {
    2: "hola mundo",
    3: "buenos dias",
    4: "feliz lunes",
    5: "hasta luego",
    6: "hola mundo cruel",
    7: "buenas noches",
}


def _synthetic_events() -> pd.DataFrame:
    rows: list[dict] = []
    subjects = ["Pinet2024Meg/S1", "Pinet2024Meg/S2"]
    t = 0.0
    for subj in subjects:
        timeline = f"{subj}_ses1_block1".replace("/", "_")
        trials = [(0, "practice uno"), (1, "practice dos")] + list(_SENTENCES.items())
        for trial_id, sent in trials:
            for ch in sent.replace(" ", "&"):
                rows.append(
                    dict(
                        type="Button",
                        trial_id=float(trial_id),
                        timeline=timeline,
                        subject=subj,
                        button=ch,
                        sentence=sent,
                        text=ch,
                        start=t,
                        duration=0.1,
                        stop=t + 0.1,
                    )
                )
                t += 0.15
    return pd.DataFrame(rows)


def test_spanish_preprocessing():
    """Check the core invariants of the SpanishBCBL preprocessing transform.

    Why it matters: this transform rewrites raw events into the layout the rest of
    the pipeline assumes. The asserts pin the contract: Button events become
    Keystroke, a Sentence event is synthesised per group, the two practice trials
    (ids 0/1) are dropped, the metadata columns the model/scripts need exist,
    subjects are factorised to integers, and there is exactly one Sentence per UID.
    """
    out = SpanishBCBLPreprocessing().run(_synthetic_events())

    types = set(out["type"].unique())
    assert "Keystroke" in types
    assert "Sentence" in types
    assert "Button" not in types  # raw "Button" must be renamed to "Keystroke"

    assert not out["trial_id"].isin([0.0, 1.0]).any()  # practice trials removed
    for col in ("sentence_UID", "sentence_typed", "button_UID"):
        assert col in out.columns

    ks = out[out["type"] == "Keystroke"]
    assert {int(s) for s in ks["subject"].unique()} == {0, 1}  # factorised ids

    sents = out[out["type"] == "Sentence"]
    assert len(sents) == sents["sentence_UID"].nunique()  # one Sentence per UID


def test_v1_splitter_is_deterministic():
    """The train/val/test splitter is reproducible for a fixed seed.

    Why it matters: the splitter clusters paraphrase-similar sentences (TF-IDF
    cosine) and assigns whole clusters to a split to prevent train/test leakage.
    A non-deterministic split would make reported numbers irreproducible, so we
    run it twice and require an identical per-keystroke split assignment.
    """
    events = SpanishBCBLPreprocessing().run(_synthetic_events())
    out = Brain2QwertyV1Splitter(seed=1).run(events)

    assert "split" in out.columns
    ks = out[out["type"] == "Keystroke"]
    assert ks["split"].notna().all()
    assert set(ks["split"].unique()).issubset({"train", "val", "test"})

    # same seed + same input -> byte-identical split assignment
    out2 = Brain2QwertyV1Splitter(seed=1).run(
        SpanishBCBLPreprocessing().run(_synthetic_events())
    )
    pd.testing.assert_series_equal(
        out.sort_values("button_UID")["split"].reset_index(drop=True),
        out2.sort_values("button_UID")["split"].reset_index(drop=True),
    )
