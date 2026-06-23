# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Brain2Qwerty V2 - sentence-level end-to-end decoder (CTC + contrastive + LLM)."""

import os

# Reduce CUDA fragmentation OOM in the long, memory-heavy LLM phase; set here so
# it applies to every entry point before torch initialises CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
