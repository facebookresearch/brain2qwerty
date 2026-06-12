# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pydantic configurations for metrics."""

import typing as tp
from inspect import isclass

import pydantic
import torch.nn as nn
from neuraltrain.metrics import metrics
from neuraltrain.utils import all_subclasses, convert_to_pydantic
from torchmetrics import Metric

from neuralset.infra import helpers

custom_metrics = [
    obj for obj in metrics.__dict__.values() if isclass(obj) and issubclass(obj, Metric)
]


TORCHMETRICS_NAMES = {
    metric_class.__name__: metric_class
    for metric_class in all_subclasses(Metric)
    if metric_class not in custom_metrics
}


class BaseMetricConfig(pydantic.BaseModel):
    """Base class for loss configurations."""

    model_config = pydantic.ConfigDict(extra="forbid")

    log_name: str
    name: str

    def build(self) -> nn.Module:
        raise NotImplementedError


for metric_class in custom_metrics:
    metric_class_name = metric_class.__name__
    config_cls = convert_to_pydantic(
        metric_class,
        metric_class_name,
        parent_class=BaseMetricConfig,
        exclude_from_build=["log_name"],
    )
    locals()[f"{metric_class_name}Config"] = config_cls


class TorchMetricConfig(BaseMetricConfig):
    name: tp.Literal[tuple(TORCHMETRICS_NAMES.keys())]  # type: ignore
    kwargs: dict[str, tp.Any] = {}

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        # validation of mandatory/extra args + basic types (str/int/float)
        helpers.validate_kwargs(TORCHMETRICS_NAMES[self.name], self.kwargs)

    def build(self) -> nn.Module:
        return TORCHMETRICS_NAMES[self.name](**self.kwargs)  # type: ignore
