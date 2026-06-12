# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pre-compute and cache M/EEG features for all subjects.

Run once before training to avoid reprocessing raw data on every run:

    python -m brain2qwerty.compute_cache --modality meg
    python -m brain2qwerty.compute_cache --modality eeg
"""

import argparse
import os

import neuralset as ns
from neuralset.data import StudyLoader


def main():
    parser = argparse.ArgumentParser(description="Cache preprocessed M/EEG features")
    parser.add_argument(
        "--modality",
        choices=["meg", "eeg"],
        default="meg",
        help="Recording modality (default: meg)",
    )
    args = parser.parse_args()

    _brainai_root = os.environ.get("BRAINAI_ROOT", os.path.expanduser("~/brainai"))
    _data_root = os.environ.get("BRAINAI_DATA_ROOT", os.path.join(_brainai_root, "data"))
    cache = os.environ.get(
        "BRAINAI_CACHE",
        os.path.join(_brainai_root, "cache", "brain2qwerty"),
    )

    if args.modality == "meg":
        study_name, feature_cls = "Pinet2024Meg", ns.features.Meg
    else:
        study_name, feature_cls = "Pinet2024Eeg", ns.features.Eeg

    study = StudyLoader(
        name=study_name,
        path=os.environ.get("BRAINAI_STUDIES_PATH", os.path.join(_data_root, "studies")),
        query=None,
        infra={"folder": cache, "mode": "cached"},
    )
    events = study.build()

    neuro = feature_cls(
        frequency=50.0,
        filter=(0.1, 20),
        baseline=(0.0, 0.2),
        scaler="RobustScaler",
        clamp=5,
        infra={"keep_in_ram": True, "folder": cache, "cluster": None},
    )
    neuro.prepare(events)
    print(f"[INFO] {args.modality.upper()} cache done.")


if __name__ == "__main__":
    main()
