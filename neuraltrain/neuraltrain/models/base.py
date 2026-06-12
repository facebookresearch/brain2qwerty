# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pydantic configurations for models."""

import pydantic
import torch.nn as nn


class BaseModelConfig(pydantic.BaseModel):
    """Base class for model configurations."""

    model_config = pydantic.ConfigDict(extra="forbid")
    name: str

    def build(self, *args, **kwargs) -> nn.Module:
        raise NotImplementedError
