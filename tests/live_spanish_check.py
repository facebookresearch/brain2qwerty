# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Live integration check for the Spanish (SpanishBCBL / Pinet2024Meg) pipeline.

Requires the real dataset on disk; not part of the default pytest suite.
Run explicitly:

    python tests/live_spanish_check.py /path/to/SpanishBCBL
"""

import sys

import neuralset as ns
import studies  # noqa: F401  (registers Pinet2024Meg)
from brain2qwerty_v1.transforms import Brain2QwertyV1Splitter, SpanishBCBLPreprocessing
from brain2qwerty_v1.utils import BUTTON_MAPPING, ChannelPositions2D
from studies.spanishbcbl import Pinet2024Meg


def main(path: str) -> None:
    print("=" * 70)
    print("[1] Create study (single timeline via query)")
    study = Pinet2024Meg(
        path=path,
        query="timeline_index == 0",
        infra_timelines={"cluster": None},
    )
    print("    resolved path:", study.path)
    assert study.path.exists(), "study path does not exist"

    print("[2] Discover files (_get_all_files)")
    recs = study._get_all_files()
    print(f"    {len(recs)} recordings found; columns={list(recs.columns)}")
    assert len(recs) > 0

    print("[3] study.run() for one timeline")
    events = study.run()
    types = events["type"].value_counts().to_dict()
    print("    event types:", types)
    print("    subject values:", sorted(events["subject"].dropna().unique())[:5])
    assert any(t in types for t in ("Button", "Keystroke")), "no keystroke events"
    assert study.device in types, f"no {study.device} event"
    assert events["subject"].str.startswith("Pinet2024Meg/").any(), "subject prefix wrong"

    print("[4] Apply transforms (SpanishBCBLPreprocessing -> Brain2QwertyV1Splitter)")
    events = SpanishBCBLPreprocessing().run(events)
    events = Brain2QwertyV1Splitter(seed=1).run(events)
    print("    types after:", events["type"].value_counts().to_dict())
    print(
        "    splits:",
        events[events.type == "Keystroke"]["split"].value_counts().to_dict(),
    )

    print("[5] Build extractors + segments + pull one item")
    neuro = ns.extractors.MegExtractor(
        frequency=50,
        filter=(0.1, 20.0),
        baseline=(0.0, 0.2),
        apply_proj=False,
        clamp=5,
        scaler="RobustScaler",
    )
    extractor = ns.extractors.LabelEncoder(
        aggregation="trigger",
        predefined_mapping=BUTTON_MAPPING,
        event_types="Keystroke",
        event_field="button",
        return_one_hot=False,
    )
    neuro.prepare(events)
    extractor.prepare(events)
    subject_encoder = ns.extractors.LabelEncoder(event_types="Meg", event_field="subject")
    subject_encoder.prepare(events)
    chan_pos = ChannelPositions2D(neuro=neuro)
    chan_pos.prepare(events)

    extractors = {
        "neuros": neuro,
        "phonemes": extractor,
        "days": subject_encoder,
        "chan_pos": chan_pos,
    }
    mask = events.type == "Keystroke"
    segments = ns.segments.list_segments(events, mask, start=-0.2, duration=0.5)
    print(f"    {len(segments)} keystroke segments")
    assert len(segments) > 0

    dataset = ns.SegmentDataset(
        extractors=extractors, segments=segments, remove_incomplete_segments=True
    )
    item = dataset[0]
    shapes = {k: tuple(v.shape) for k, v in item.data.items() if hasattr(v, "shape")}
    print("    item shapes:", shapes)
    neuros = item.data["neuros"]
    print(
        "    neuros dtype:",
        neuros.dtype,
        "min/max:",
        float(neuros.min()),
        float(neuros.max()),
    )
    assert neuros.ndim == 3, "expected (1, C, T) neuros"
    assert item.data["phonemes"] is not None
    print("=" * 70)
    print("LIVE SPANISH CHECK PASSED")


if __name__ == "__main__":
    import os

    default = os.environ.get("BRAIN2QWERTY_STUDIES", "")
    main(sys.argv[1] if len(sys.argv) > 1 else default)
