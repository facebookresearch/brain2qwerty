# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from brain2qwerty.metrics import SentenceCER
from brain2qwerty.utils import NUM_CLASSES


def test_perfect_prediction():
    """CER should be 0 when predictions match targets exactly."""
    metric = SentenceCER()
    y_true = torch.tensor([0, 1, 2, 3, 8, 7])
    y_pred = torch.zeros(len(y_true), NUM_CLASSES)
    for i, t in enumerate(y_true):
        y_pred[i, t] = 10.0
    metric.update(y_pred, y_true)
    assert metric.compute().item() == 0.0


def test_fully_wrong_prediction():
    """CER should be > 0 when predictions are all wrong."""
    metric = SentenceCER()
    y_true = torch.tensor([0, 1, 2])
    y_pred = torch.zeros(len(y_true), NUM_CLASSES)
    for i in range(len(y_true)):
        wrong_class = (y_true[i].item() + 5) % NUM_CLASSES
        y_pred[i, wrong_class] = 10.0
    metric.update(y_pred, y_true)
    assert metric.compute().item() > 0.0


def test_cer_accumulates_across_updates():
    """CER should average across multiple update calls."""
    metric = SentenceCER()

    y_true1 = torch.tensor([0, 1, 2])
    y_pred1 = torch.zeros(3, NUM_CLASSES)
    for i, t in enumerate(y_true1):
        y_pred1[i, t] = 10.0
    metric.update(y_pred1, y_true1)

    y_true2 = torch.tensor([0, 1, 2])
    y_pred2 = torch.zeros(3, NUM_CLASSES)
    for i in range(3):
        y_pred2[i, (y_true2[i].item() + 5) % NUM_CLASSES] = 10.0
    metric.update(y_pred2, y_true2)

    cer = metric.compute().item()
    assert 0.0 < cer < 1.0
