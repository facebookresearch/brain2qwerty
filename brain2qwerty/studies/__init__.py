# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone study definitions for the released datasets (SpanishBCBL, EnglishBCBL).

Importing this package registers the study subclasses with neuralset so they
are reachable via ``Study(name=...)``.
"""

from . import englishbcbl as englishbcbl  # noqa: F401  (registers PinetAudio2025)
from . import spanishbcbl as spanishbcbl  # noqa: F401  (registers Pinet2024Meg)
