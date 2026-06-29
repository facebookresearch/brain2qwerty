import importlib
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch


@dataclass
class _Batch:
    data: dict
    segments: list


class _SegmentDataset:
    def __init__(self, extractors, segments, remove_incomplete_segments=False):
        self.extractors = extractors
        self.segments = segments


class _BaseExtractor:
    pass


class _MegExtractor(_BaseExtractor):
    def __init__(self, tensor):
        self.tensor = tensor
        self.frequency = 10

    def __call__(self, ns_events, start, duration, trigger):
        return self.tensor


class _DummyExtractor(_BaseExtractor):
    def __init__(self, tensor):
        self.tensor = tensor

    def __call__(self, ns_events, start, duration, trigger):
        return self.tensor


def _install_stubs(monkeypatch):
    neuralset_pkg = types.ModuleType("neuralset")
    neuralset_pkg.segments = types.SimpleNamespace(Segment=object)

    dataloader_mod = types.ModuleType("neuralset.dataloader")
    dataloader_mod.Batch = _Batch
    dataloader_mod.SegmentDataset = _SegmentDataset

    extractors_mod = types.ModuleType("neuralset.extractors")
    extractors_mod.BaseExtractor = _BaseExtractor

    neuro_mod = types.ModuleType("neuralset.extractors.neuro")
    neuro_mod.MegExtractor = _MegExtractor

    utils_mod = types.ModuleType("brain2qwerty_v2.utils")

    def apply_jitter(data, seg, feat):
        return data[:, 1:]

    utils_mod.apply_jitter = apply_jitter

    monkeypatch.setitem(sys.modules, "neuralset", neuralset_pkg)
    monkeypatch.setitem(sys.modules, "neuralset.dataloader", dataloader_mod)
    monkeypatch.setitem(sys.modules, "neuralset.extractors", extractors_mod)
    monkeypatch.setitem(sys.modules, "neuralset.extractors.neuro", neuro_mod)
    monkeypatch.setitem(sys.modules, "brain2qwerty_v2.utils", utils_mod)


def _load_data_module(monkeypatch):
    _install_stubs(monkeypatch)
    sys.modules.pop("brain2qwerty_v2.data", None)
    return importlib.import_module("brain2qwerty_v2.data")


def test_sentence_dataset_getitem_applies_jitter_to_meg_only(monkeypatch):
    data_mod = _load_data_module(monkeypatch)

    seg = SimpleNamespace(ns_events=None, start=0.0, duration=1.0, trigger=SimpleNamespace())
    ds = data_mod.SentenceDataset(
        extractors={
            "neuros": _MegExtractor(torch.arange(12, dtype=torch.float32).reshape(2, 6)),
            "phonemes": _DummyExtractor(torch.tensor([1, 2, 3], dtype=torch.long)),
        },
        segments=[seg],
        jitter=True,
    )

    item = ds[0]
    assert item.data["neuros"].shape == (1, 2, 5)
    assert item.data["phonemes"].shape == (1, 3)


def test_sentence_dataset_getitem_requires_int_index(monkeypatch):
    data_mod = _load_data_module(monkeypatch)
    ds = data_mod.SentenceDataset(extractors={}, segments=[])

    with pytest.raises(ValueError):
        _ = ds["0"]


def test_sentence_dataset_collate_fn_pads_variable_lengths(monkeypatch):
    data_mod = _load_data_module(monkeypatch)
    ds = data_mod.SentenceDataset(extractors={}, segments=[])

    b1 = _Batch(
        data={
            "phonemes": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "neuros": torch.tensor([[[1.0, 2.0, 3.0], [10.0, 11.0, 12.0]]]),
            "days": torch.tensor(1),
            "chan_pos": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]]),
        },
        segments=[SimpleNamespace(id=1)],
    )
    b2 = _Batch(
        data={
            "phonemes": torch.tensor([[4, 5]], dtype=torch.long),
            "neuros": torch.tensor([[[5.0, 6.0], [15.0, 16.0]]]),
            "days": torch.tensor(2),
            "chan_pos": torch.tensor([[[0.5, 0.6], [0.7, 0.8]]]),
        },
        segments=[SimpleNamespace(id=2)],
    )

    out = ds.collate_fn([b1, b2])

    assert tuple(out.data["phoneme_sizes"].tolist()) == (3, 2)
    assert out.data["phonemes"].shape == (2, 3)
    assert tuple(out.data["neuro_sizes"].tolist()) == (3, 2)
    assert out.data["neuros"].shape == (2, 3, 2)
    assert tuple(out.data["days"].tolist()) == (1, 2)
    assert out.data["chan_pos"].shape == (2, 2, 2)
    assert [s.id for s in out.segments] == [1, 2]
