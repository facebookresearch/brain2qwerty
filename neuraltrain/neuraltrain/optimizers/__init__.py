# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import pydantic

from ..utils import all_subclasses
from .base import BaseLRSchedulerConfig, BaseOptimizerConfig, LightningOptimizerConfig

OptimizerConfig = BaseOptimizerConfig  # so that OptimizerConfig is defined for mypy


def update_config_optimizer() -> None:
    global OptimizerConfig  # pylint: disable=global-statement
    from .base import BaseOptimizerConfig

    OptimizerConfig = tp.Annotated[  # type: ignore
        tp.Union[tuple(all_subclasses(BaseOptimizerConfig))],
        pydantic.Field(discriminator="name"),  # serves for pydantic
    ]


update_config_optimizer()
