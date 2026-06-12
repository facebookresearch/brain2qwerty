# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from brain2qwerty.utils import BUTTON_MAPPING, CHAR_INDEX, NUM_CLASSES


def test_button_mapping_consistency():
    """CHAR_INDEX should be the inverse of the canonical 29 classes."""
    assert NUM_CLASSES == 29
    assert len(CHAR_INDEX) == NUM_CLASSES
    for idx, char in CHAR_INDEX.items():
        assert 0 <= idx < NUM_CLASSES


def test_button_mapping_special_chars():
    """Special characters should all map to class 13."""
    for char in ["ý", "ü", "û", "£", "¤", "-", "¿", "`"]:
        assert BUTTON_MAPPING[char] == 13
