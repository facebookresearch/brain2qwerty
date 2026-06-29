# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from collections import defaultdict

import torch
from torch.utils.data import Sampler

from neuralset.extractors.base import BaseStatic
from neuralset.extractors.neuro import ChannelPositions as _ChannelPositions

BUTTON_MAPPING = {
    "s": 0,
    "o": 1,
    "t": 2,
    "e": 3,
    "n": 4,
    "c": 5,
    "i": 6,
    "a": 7,
    "<space>": 8,
    "d": 9,
    "l": 10,
    "r": 11,
    "b": 12,
    "<special>": 13,
    "z": 14,
    "v": 15,
    "f": 16,
    "m": 17,
    "u": 18,
    "h": 19,
    "p": 20,
    "g": 21,
    "q": 22,
    "w": 23,
    "x": 24,
    "y": 25,
    "j": 26,
    "k": 27,
    "<number>": 28,
    "ý": 13,
    "\x14": 13,
    "ü": 13,
    "û": 13,
    "£": 13,
    "¤": 13,
    "-": 13,
    "¿": 13,
    "`": 13,
}

NUM_CLASSES = len(set(BUTTON_MAPPING.values()))

CHAR_INDEX = {
    0: "s",
    1: "o",
    2: "t",
    3: "e",
    4: "n",
    5: "c",
    6: "i",
    7: "a",
    8: " ",
    9: "d",
    10: "l",
    11: "r",
    12: "b",
    13: "@",
    14: "z",
    15: "v",
    16: "f",
    17: "m",
    18: "u",
    19: "h",
    20: "p",
    21: "g",
    22: "q",
    23: "w",
    24: "x",
    25: "y",
    26: "j",
    27: "k",
    28: "9",
}

NGRAM_CHAR_INDEX = {**CHAR_INDEX, 8: "&"}


# --- Channel positions (paper's 2D MEG layout) -----------------------------
class ChannelPositions2D(_ChannelPositions):
    """Re-enable 2D channel positions for MEG to match the paper."""

    def model_post_init(self, log__: tp.Any) -> None:
        BaseStatic.model_post_init(self, log__)
        if self.neuro is not None:
            if self.event_types not in {"MneRaw", self.neuro.event_types}:
                raise ValueError(
                    f"event_types={self.event_types} must match "
                    f"neuro.event_types={self.neuro.event_types}."
                )
            self._neuro = self.neuro


# --- DDP sampler (keeps a sentence's keystrokes on one rank) ---------------
class SentenceGroupedDistributedSampler(Sampler):
    """Distribute whole sentences to DDP ranks.

    All keystrokes of a sentence stay on the same rank so the sentence-level
    transformer always sees complete sentences. Ranks are padded with repeated
    samples to equal length so DDP collectives stay synchronised.
    """

    def __init__(self, segments, seed: int = 0, shuffle: bool = False) -> None:
        self.seed = seed
        self.epoch = 0
        self.shuffle = shuffle
        groups: dict[tp.Any, list[int]] = defaultdict(list)
        for idx, seg in enumerate(segments):
            uid = seg.trigger.extra.get("sentence_UID", str(idx))
            groups[uid].append(idx)
        self._groups = list(groups.values())

    def _rank_world(self) -> tuple[int, int]:
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def _distribute(self) -> tuple[list[list[int]], int]:
        _, world_size = self._rank_world()
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        order = (
            torch.randperm(len(self._groups), generator=g).tolist()
            if self.shuffle
            else list(range(len(self._groups)))
        )
        per_rank: list[list[int]] = [[] for _ in range(world_size)]
        for i, gi in enumerate(order):
            per_rank[i % world_size].extend(self._groups[gi])
        max_len = max((len(r) for r in per_rank), default=0)
        for r in per_rank:
            deficit = max_len - len(r)
            if deficit > 0 and r:
                r.extend((r * (deficit // len(r) + 1))[:deficit])
        return per_rank, max_len

    def __iter__(self):
        rank, _ = self._rank_world()
        per_rank, _ = self._distribute()
        return iter(per_rank[rank])

    def __len__(self) -> int:
        _, max_len = self._distribute()
        return max_len

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


# SpanishBCBL participant bookkeeping (single source of truth for the training
# preprocessing): recover the 19 unique MEG participants by dropping the
# no-keyboard controls and one excluded subject (metallic implant), then merging
# recording ids that belong to the same person.
CONTROL_SUBJECTS = {
    "Pinet2024Meg/S11122024",
    "Pinet2024Meg/S12122024",
    "Pinet2024Meg/S26112024",
    "Pinet2024Meg/S27112024",
    "Pinet2024Meg/S28112024",
}
EXCLUDED_SUBJECTS = {"Pinet2024Meg/S23"}
SUBJECT_MERGE = {
    "Pinet2024Meg/S18": "Pinet2024Meg/S1",
    "Pinet2024Meg/S14": "Pinet2024Meg/S4",
    "Pinet2024Meg/S10": "Pinet2024Meg/S5",
    "Pinet2024Meg/S21": "Pinet2024Meg/S5",
}


def select_participants(events):
    """Keep the 19 unique SpanishBCBL participants: drop control/excluded subjects
    and merge duplicate recordings (the ``subject`` column stays a string)."""
    keep = ~events["subject"].isin(CONTROL_SUBJECTS | EXCLUDED_SUBJECTS)
    events = events[keep].copy()
    events["subject"] = events["subject"].replace(SUBJECT_MERGE)
    return events


# --- Training helpers ------------------------------------------------------
def materialize_lazy_params(module: torch.nn.Module, loader) -> None:
    """Initialise lazy params (e.g. the channel merger's ``LazyLinear``) with a
    dummy forward, so the model has no uninitialised parameters before DDP wraps
    it; any params still uninitialised afterwards are replaced with empty ones."""
    batch = next(iter(loader))
    module.eval()
    with torch.no_grad():
        module(batch)
    module.train()
    for sub in module.modules():
        for name, param in list(getattr(sub, "_parameters", {}).items()):
            if isinstance(param, torch.nn.UninitializedParameter):
                sub._parameters[name] = torch.nn.Parameter(torch.empty(1))
