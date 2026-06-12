# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=super-init-not-called

import collections
import dataclasses
import logging
import typing as tp
import warnings

import torch

import neuralset as ns

from .base import Frequency
from .features import BaseFeature as Feat
from .segments import find_incomplete_segments

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SegmentData:
    """Dataclass with fields:
    - segments: the list of segments corresponding to this data
    - data: a dictionary of extracted features. The data is always batched on 1st dim"
    """

    data: tp.Dict[str, torch.Tensor]
    segments: tp.List[ns.segments.Segment]

    def __post_init__(self) -> None:
        if not isinstance(self.data, dict):
            raise TypeError(f"'features' need to be a dict, got: {self.features}")
        if not self.data:
            raise ValueError(f"No data in {self}")
        if not isinstance(self.segments, list):
            raise TypeError(f"'segments' needs to be a list, got {self.segments}")
        # check batch dimension
        batch_size = next(iter(self.data.values())).shape[0]
        if len(self.segments) != batch_size:
            raise RuntimeError(
                f"Incoherent batch size {batch_size} for {len(self.segments)} segments in {self}"
            )

    def to(self, device: str) -> "SegmentData":
        """Creates a new instance on the appropriate device"""
        out = {name: d.to(device) for name, d in self.data.items()}
        return SegmentData(data=out, segments=self.segments)

    # pylint: disable=unused-argument
    def __getitem__(self, key: str) -> None:
        raise RuntimeError("New SegmentData batch is not a dict, use batch.data instead")


def validate_features(features: tp.Mapping[str, Feat]) -> tp.Mapping[str, Feat]:
    """Validate the feature container provided as input
    and map all cases to the more general a dict of list of sequences of features
    """
    if not features:
        return {}
    # use feature names for list
    if not isinstance(features, collections.abc.Mapping):
        raise ValueError(f"Only dict of features are supported, got {type(features)}")
    # single features are mapped to list to unify all cases
    return features


def _get_pad_lengths(
    feats: tp.Mapping[str, Feat],
    pad_duration: float | None,  # in seconds
) -> tp.Dict[str, int]:
    """Precompute pad length in samples for each feature if applicable
    feats: mapping of Features
        the features
    pad_duration: float or None
        padding duration in seconds (if any)
    """
    pad_lengths: tp.Dict[str, int] = {}
    if pad_duration is None:
        return pad_lengths
    for name, f in feats.items():
        # You don't want to apply padding to channel positions and subject ids
        if name not in ["neuro", "feature"]:
            continue

        if isinstance(f, Feat):
            freq = Frequency(f.frequency)
            pad_lengths[name] = freq.to_ind(pad_duration)
    return pad_lengths


def _pad_to(tensor: torch.Tensor, pad_len: int | None):
    """Pad last dimension to a given length"""
    if pad_len is None:
        return tensor
    if pad_len < tensor.shape[-1]:
        msg = "Pad duration is shorter than segment duration, cropping."
        warnings.warn(msg, UserWarning)
        return tensor[:, :pad_len]
    else:
        return torch.nn.functional.pad(tensor, (0, pad_len - tensor.shape[-1]))


def _apply_feature(segment: ns.segments.Segment, feature: Feat) -> torch.Tensor:
    """Apply feature on a segment"""
    return feature(
        segment.ns_events,
        start=segment.start,
        duration=segment.duration,
        trigger=segment._trigger,
    )


class SegmentDataset(torch.utils.data.Dataset[SegmentData]):
    """Dataset defined through segments and features

    Parameters
    ----------
    features: dict of Feature
        features to be computed, returned in the SegmentData.data dictionary items
    segments: list of segments
        the list of ns.segments.Segment instances defining the dataset
    pad_duration: float or None
        pad to a given duration
    remove_incomplete_segments: bool
         remove segments which do not contain events for one of the features

    Usage
    -----
    .. code-block:: python

        feats = {"whatever": ns.features.Pulse(frequency=100.0)]}
        ds = dl.SegmentDataset(feats, segments)
        # one data item
        item = ds[0]
        assert item.data["whatever"].shape[0] == 1  # batch dimension is always added
        # through dataloader:
        dataloader = torch.utils.data.DataLoader(ds, collate_fn=ds.collate_fn, batch_size=2)
        batch = next(iter(dataloader))
        print(batch.data["whatever"])
        # batch.segments holds the corresponding segments

    """

    def __init__(
        self,
        features: tp.Mapping[str, Feat],
        segments: tp.Sequence[ns.segments.Segment],
        pad_duration: float | None = None,
        remove_incomplete_segments: bool = False,
    ) -> None:
        self.features = validate_features(features)
        self.pad_duration = pad_duration
        self.segments = segments
        self._pad_lengths: None | dict[str, int] = None
        self.remove_incomplete_segments = remove_incomplete_segments
        self.check_incomplete_segments(remove=remove_incomplete_segments)

    def check_incomplete_segments(self, remove: bool = False) -> None:
        event_types = {
            e
            for f in self.features.values()
            for e in f._event_types_helper.classes
            if f.aggregation != "trigger"
        }
        invalid_indices = find_incomplete_segments(self.segments, list(event_types))
        if invalid_indices:
            no_default = [f for f in self.features.values() if f._missing_default is None]
            if remove:
                msg = f"Removing {len(invalid_indices)} segments out of {len(self.segments)}"
                logger.info(msg)
                self.segments = [
                    s for i, s in enumerate(self.segments) if i not in invalid_indices
                ]
            elif no_default:
                names = {m.__class__.__name__ for m in no_default}
                msg = f"{len(invalid_indices)} segments are missing some event types. Use `remove_incomplete_segments=True` to remove them,"
                msg += f" or populate the _missing_default of features {names} by calling them once before."
                raise ValueError(msg)
            else:
                msg = f"Keeping {len(invalid_indices)} segments out of {len(self.segments)} "
                msg += "as all features have there shape populated"
                logger.info(msg)

    def collate_fn(self, batches: tp.List[SegmentData]) -> SegmentData:
        """Creates a new instance from several by stacking in a new first dimension
        for all attributes
        """
        if not batches:
            return SegmentData(data={}, segments=[])
        if len(batches) == 1:
            return batches[0]
        if not batches[0].data:
            raise ValueError(f"No feature in first batch: {batches[0]}")
        # move everything to pytorch if first one is numpy
        features = {}
        for name in batches[0].data:
            data = [b.data[name] for b in batches]
            try:
                features[name] = torch.cat(data, axis=0)  # type: ignore
            except Exception:
                string = f"Failed to collate data with shapes {[d.shape for d in data]}\n"
                string += "Do you need specifying padding in SegmentDataset?"
                logger.warning(string)
                raise
        segments = [s for b in batches for s in b.segments]
        return SegmentData(data=features, segments=segments)

    def _check_padding(self) -> None:  # check if padding is needed
        if self._pad_lengths is not None:
            return
        if self.pad_duration is None:
            if len(set([s.duration for s in self.segments])) > 1:
                msg = "Segments have different durations, so they cannot be collated into batches."
                msg += " Set `pad_duration` to `auto` to pad the segments to the maximum duration."
                raise ValueError(msg)
            pad_duration = self.pad_duration
        elif self.pad_duration == "auto":
            pad_duration = max([s.duration for s in self.segments])
        else:
            pad_duration = self.pad_duration
        self._pad_lengths = _get_pad_lengths(self.features, pad_duration)

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> SegmentData:
        self._check_padding()
        assert isinstance(self._pad_lengths, dict)  # for mpy
        seg = self.segments[idx]
        out: tp.Dict[str, torch.Tensor] = {}
        for name, feats in self.features.items():
            # get data, merge if need be
            data = _apply_feature(seg, feats)
            # pad if need be
            data = _pad_to(data, self._pad_lengths.get(name, None))
            # append to specific feature list
            out[name] = data[None, ...]  # add back dimension and set
        return SegmentData(data=out, segments=[seg])

    def build_dataloader(self, **kwargs: tp.Any) -> torch.utils.data.DataLoader:
        """Returns a dataloader for this dataset"""
        return torch.utils.data.DataLoader(self, collate_fn=self.collate_fn, **kwargs)

    def as_one_batch(self, num_workers: int = 0) -> SegmentData:
        """Returns a single batch with all the dataset data, un-shuffled"""
        num_workers = min(num_workers, len(self))
        batch_size = len(self)
        if num_workers > 1:
            batch_size = max(1, len(self) // (3 * num_workers))
        if num_workers == 1:
            num_workers = 0  # simplifies debugging
        loader = self.build_dataloader(
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=False,
        )
        return self.collate_fn(list(loader))
