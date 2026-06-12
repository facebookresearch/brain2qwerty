# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared test fixtures producing synthetic data that mimics the real
Pinet2024 MEG/EEG structure, so tests run without the actual dataset."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

from brain2qwerty.utils import NUM_CLASSES

SMALL_HIDDEN = 128
SMALL_CHANNELS = 32
SMALL_FOURIER_DIM = 32


@dataclass
class FakeSegment:
    _trigger: dict = field(default_factory=dict)


def make_fake_batch(
    n_samples: int = 16,
    n_channels: int = SMALL_CHANNELS,
    n_timepoints: int = 25,
    n_classes: int = NUM_CLASSES,
    n_sentences: int = 4,
    device: str = "cpu",
):
    """Build a synthetic SegmentData-like object for testing."""
    neuro = torch.randn(n_samples, n_channels, n_timepoints, device=device)
    feature = torch.randint(0, n_classes, (n_samples, 1), device=device)
    subject_id = torch.zeros(n_samples, dtype=torch.long, device=device)
    channel_positions = torch.randn(n_samples, n_channels, 2, device=device)

    segments = []
    samples_per_sentence = max(1, n_samples // n_sentences)
    for i in range(n_samples):
        sent_idx = i // samples_per_sentence
        segments.append(
            FakeSegment(
                _trigger={
                    "trial_id": float(sent_idx),
                    "timeline": "fake_subject-S1_session-1_task-block1",
                    "sentence": "el gato come pescado",
                    "button_unique_id": f"sent{sent_idx}_{i}",
                    "sequence_id": sent_idx,
                }
            )
        )

    data = {
        "neuro": neuro,
        "feature": feature,
        "subject_id": subject_id,
        "channel_positions": channel_positions,
    }
    return SimpleNamespace(data=data, segments=segments)


@pytest.fixture
def brain_model():
    """Minimal SimpleConvTimeAgg built via the config system."""
    from neuraltrain.models.simpleconv import SimpleConvTimeAggConfig

    config = SimpleConvTimeAggConfig(
        name="SimpleConvTimeAgg",
        hidden=SMALL_HIDDEN,
        depth=2,
        kernel_size=3,
        dilation_period=1,
        time_agg_out="att",
        dropout_input=0.0,
        conv_dropout=0.0,
        batch_norm=True,
        initial_linear=64,
        gelu=True,
        skip=True,
        scale=0.1,
        relu_leakiness=0.01,
        subject_layers_config={},
        merger_config={
            "n_virtual_channels": 16,
            "fourier_emb_config": {
                "n_freqs": None,
                "total_dim": SMALL_FOURIER_DIM,
                "n_dims": 2,
            },
            "dropout": 0.0,
            "usage_penalty": 1.0,
            "per_subject": True,
            "embed_ref": False,
        },
    )
    return config.build(n_in_channels=SMALL_CHANNELS, n_outputs=SMALL_HIDDEN)


@pytest.fixture
def transformer_model():
    """Minimal TransformerEncoder for CPU testing."""
    from neuraltrain.models.transformer import TransformerEncoderConfig

    config = TransformerEncoderConfig(
        name="TransformerEncoder",
        depth=1,
        heads=1,
        alibi_pos_bias=True,
    )
    return config.build(dim=SMALL_HIDDEN)


@pytest.fixture
def fake_batch():
    return make_fake_batch()
