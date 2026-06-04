# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import warnings

import pydantic

from .base import BaseFeature, BaseStatic, LabelEncoder
from .neuro import *  # noqa

FeatureConfig = BaseFeature


def update_config_feature() -> None:
    global FeatureConfig  # pylint: disable=global-statement
    from .base import BaseFeature

    FeatureConfig = tp.Annotated[  # type: ignore
        tp.Union[tuple(x for x in BaseFeature._CLASSES.values())],
        pydantic.Field(discriminator="name"),  # serves for pydantic
    ]


update_config_feature()


def __getattr__(name: str) -> tp.Any:
    if name == "CfgFeature":
        warnings.warn("CfgFeature is replaced by FeatureConfig", DeprecationWarning)
        return FeatureConfig
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
