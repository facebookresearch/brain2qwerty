# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import pydantic

from ..utils import all_subclasses
from .base import BaseModelConfig
from .simpleconv import (
    SimpleConv,
    SimpleConvConfig,
    SimpleConvTimeAgg,
    SimpleConvTimeAggConfig,
)
from .transformer import TransformerEncoderConfig

ModelConfig = BaseModelConfig  # so that ModelConfig is defined for mypy


def update_config_model() -> None:
    global ModelConfig  # pylint: disable=global-statement
    from .base import BaseModelConfig

    ModelConfig = tp.Annotated[  # type: ignore
        tp.Union[tuple(all_subclasses(BaseModelConfig))],
        pydantic.Field(discriminator="name"),  # serves for pydantic
    ]


update_config_model()
