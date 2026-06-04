# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Linear baseline: per-subject Ridge classifier at a fixed time sample.

Usage:
    python brain2qwerty/linear_model.py
"""

import os

import neuralset as ns
import numpy as np
from neuralset.data import StudyLoader
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .utils import BUTTON_MAPPING

PROJECT_NAME = "brain2qwerty"
_BRAINAI_ROOT = os.environ.get("BRAINAI_ROOT", os.path.expanduser("~/brainai"))
_DATA_ROOT = os.environ.get("BRAINAI_DATA_ROOT", os.path.join(_BRAINAI_ROOT, "data"))
CACHE = os.environ.get(
    "BRAINAI_CACHE", os.path.join(_BRAINAI_ROOT, "cache", PROJECT_NAME)
)
MODALITY = "Meg"

MERGE = {
    "Pinet2024Meg/S18": "Pinet2024Meg/S1",
    "Pinet2024Meg/S14": "Pinet2024Meg/S4",
    "Pinet2024Meg/S10": "Pinet2024Meg/S5",
    "Pinet2024Meg/S21": "Pinet2024Meg/S5",
}
EXCLUDE = {"Pinet2024Meg/S23"}
CONTROL = {
    "Pinet2024Meg/S11122024",
    "Pinet2024Meg/S12122024",
    "Pinet2024Meg/S26112024",
    "Pinet2024Meg/S27112024",
    "Pinet2024Meg/S28112024",
}


def apply_subject_merge(events):
    """Apply subject merging to get n=19 MEG participants."""
    events = events[~events["subject"].isin(CONTROL)]
    events = events[~events["subject"].isin(EXCLUDE)]
    events["subject"] = events["subject"].replace(MERGE)
    return events


if __name__ == "__main__":
    events = StudyLoader(
        name=f"Pinet2024{MODALITY}",
        path=os.environ.get("BRAINAI_STUDIES_PATH", os.path.join(_DATA_ROOT, "studies")),
        infra={"folder": CACHE},
    ).build()

    events = apply_subject_merge(events)
    print(f"n = {events['subject'].nunique()} subjects after merging")

    neuro = ns.features.Meg(
        frequency=100,
        filter=(0.1, 40.0),
        baseline=(0.0, 0.5),
        infra={"folder": CACHE, "cluster": None},
        scaler="StandardScaler",
    )
    button = ns.features.LabelEncoder(
        predefined_mapping=BUTTON_MAPPING,
        aggregation="trigger",
        event_types="Button",
        event_field="button",
        return_one_hot=False,
    )
    neuro.prepare(events)
    button.prepare(events)
    features = {"neuro": neuro, "button": button}

    # Decode at a fixed time index (t=54 corresponds to ~40ms post-keystroke
    # at 100 Hz sampling, where motor-related activity peaks).
    T_DECODE = 54
    results = {}

    for subject in sorted(events["subject"].unique()):
        print(f"\nSubject: {subject}")
        segments = ns.segments.list_segments(
            events,
            (events["type"] == "Button") & (events["subject"] == subject),
            duration=1.0,
            start=-0.5,
        )
        dataset = ns.SegmentDataset(features, segments)
        loader = DataLoader(
            dataset,
            batch_size=10**9,
            num_workers=4,
            collate_fn=dataset.collate_fn,
        )
        batch = next(iter(loader))

        sentence_uids = np.array(
            [
                f"{seg._trigger['trial_id']}_{seg._trigger['timeline']}"
                for seg in batch.segments
            ]
        )
        X = batch.data["neuro"].numpy().astype(np.float32)
        y = batch.data["button"].numpy().astype(np.int64).flatten()

        unique_sents = np.unique(sentence_uids)
        train_s, test_s = train_test_split(unique_sents, test_size=0.2, random_state=42)
        train_mask = np.isin(sentence_uids, train_s)
        test_mask = np.isin(sentence_uids, test_s)

        clf = RidgeClassifierCV(alphas=np.logspace(-2, 8, 11), store_cv_values=True)
        clf.fit(X[train_mask, :, T_DECODE], y[train_mask])
        y_pred = clf.predict(X[test_mask, :, T_DECODE])

        acc = (y_pred == y[test_mask]).mean()
        print(f"  Accuracy at t={T_DECODE}: {acc:.3f}")
        results[subject] = {"accuracy": float(acc)}

    print("\n--- Summary ---")
    accs = [r["accuracy"] for r in results.values()]
    print(f"Mean accuracy: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
