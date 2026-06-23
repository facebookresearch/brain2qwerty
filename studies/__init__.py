# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone study definition for the released dataset (SpanishBCBL).

Importing this package registers the study subclass with neuralset so it is
reachable via ``Study(name=...)``.
"""

from . import spanishbcbl as spanishbcbl  # noqa: F401  (registers Pinet2024Meg)
