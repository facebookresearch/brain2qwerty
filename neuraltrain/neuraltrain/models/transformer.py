# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Wrappers around Transformer models from x_transformers package.
"""

import logging
import typing as tp

import torch.nn as nn

from .base import BaseModelConfig

logger = logging.getLogger(__name__)


class TransformerEncoderConfig(BaseModelConfig):
    name: tp.Literal["TransformerEncoder"] = "TransformerEncoder"
    heads: int = 8
    depth: int = 12

    # Attention blocks
    attn_flash: bool = (
        False  # Use Flash Attention; not compatible with ALiBi and probably other features
    )
    attn_dropout: float = 0.1

    # Feedforward blocks
    ff_mult: int = 4  # Feedforward expansion factor
    ff_dropout: float = 0.0

    # Normalization
    use_scalenorm: bool = True
    use_rmsnorm: bool = False

    # Positional embedding
    rel_pos_bias: bool = False
    alibi_pos_bias: bool = False
    rotary_pos_emb: bool = True
    rotary_xpos: bool = False

    # Others
    residual_attn: bool = False
    scale_residual: bool = True
    layer_dropout: float = 0.0

    def build(self, dim: int) -> nn.Module:
        from x_transformers import Encoder  # type: ignore

        if dim % self.heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by the number of heads ({self.heads})"
            )
        kwargs = self.model_dump()
        del kwargs["name"]
        return Encoder(dim=dim, **kwargs)
