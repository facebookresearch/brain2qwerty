# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import warnings

import pydantic

from ..utils import all_subclasses
from .base import BaseMetricConfig

MetricConfig = BaseMetricConfig  # pylint: disable=global-statement


def update_config_metric() -> None:
    global MetricConfig  # pylint: disable=global-statement
    from .base import BaseMetricConfig

    MetricConfig = tp.Annotated[  # type: ignore
        tp.Union[tuple(all_subclasses(BaseMetricConfig))],
        pydantic.Field(discriminator="name"),  # serves for pydantic
    ]


update_config_metric()
