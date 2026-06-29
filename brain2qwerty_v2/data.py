# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import torch
from torch.nn.utils.rnn import pad_sequence

import neuralset as ns
from neuralset.dataloader import Batch, SegmentDataset
from neuralset.extractors import BaseExtractor
from neuralset.extractors.neuro import MegExtractor

from .utils import apply_jitter


class SentenceDataset(SegmentDataset):
    """Sentence-mode dataset with per-item MEG onset jitter (train only) and
    padded collation of variable-length sentences."""

    def __init__(
        self,
        extractors: tp.Mapping[str, BaseExtractor],
        segments: tp.Sequence[ns.segments.Segment],
        jitter: bool = False,
        *,
        remove_incomplete_segments: bool = False,
    ) -> None:
        super().__init__(
            extractors=extractors,
            segments=segments,
            remove_incomplete_segments=remove_incomplete_segments,
        )
        self.jitter = jitter

    def __getitem__(self, idx: int) -> Batch:
        if not isinstance(idx, int):
            raise ValueError(f"idx must be int, got {type(idx)}")
        seg = self.segments[idx]
        out: dict[str, torch.Tensor] = {}
        for name, extractor in self.extractors.items():
            data = extractor(
                seg.ns_events, start=seg.start, duration=seg.duration, trigger=seg.trigger
            )
            if self.jitter and isinstance(extractor, MegExtractor):
                data = apply_jitter(data, seg, extractor)
            out[name] = data[None, ...]
        return Batch(data=out, segments=[seg])

    def collate_fn(self, batches: list[Batch]) -> Batch:
        if not batches or not batches[0].data:
            return Batch(data={}, segments=[])
        batch_data = [b.data for b in batches]
        out: dict[str, tp.Any] = {}

        phonemes = [d["phonemes"].squeeze() for d in batch_data]
        if phonemes[0].ndim == 0:
            phonemes = [p.unsqueeze(0) for p in phonemes]
        out["phoneme_sizes"] = torch.tensor(
            [p.shape[0] for p in phonemes], dtype=torch.long
        )
        out["phonemes"] = pad_sequence(phonemes, batch_first=True, padding_value=0)

        neuros = [d["neuros"].squeeze().T for d in batch_data]  # -> (T, C)
        out["neuro_sizes"] = torch.tensor([n.shape[0] for n in neuros], dtype=torch.long)
        out["neuros"] = pad_sequence(neuros, batch_first=True, padding_value=0)

        out["days"] = torch.tensor([d["days"] for d in batch_data])
        out["chan_pos"] = torch.cat([d["chan_pos"] for d in batch_data])
        # sentence texts/segments stay on Batch.segments so Batch.to(device) only
        # moves tensors
        return Batch(data=out, segments=[b.segments[0] for b in batches])
