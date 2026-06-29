import importlib
import sys
import types
from types import SimpleNamespace


def _install_fake_neuralset(monkeypatch):
    neuralset_pkg = types.ModuleType("neuralset")
    extractors_pkg = types.ModuleType("neuralset.extractors")
    base_mod = types.ModuleType("neuralset.extractors.base")
    neuro_mod = types.ModuleType("neuralset.extractors.neuro")

    class BaseStatic:
        @staticmethod
        def model_post_init(*args, **kwargs):
            return None

    class ChannelPositions:
        def __init__(self, *args, **kwargs):
            self.neuro = kwargs.get("neuro")
            self.event_types = kwargs.get("event_types", "MneRaw")

    base_mod.BaseStatic = BaseStatic
    neuro_mod.ChannelPositions = ChannelPositions

    monkeypatch.setitem(sys.modules, "neuralset", neuralset_pkg)
    monkeypatch.setitem(sys.modules, "neuralset.extractors", extractors_pkg)
    monkeypatch.setitem(sys.modules, "neuralset.extractors.base", base_mod)
    monkeypatch.setitem(sys.modules, "neuralset.extractors.neuro", neuro_mod)


def _load_utils(monkeypatch):
    _install_fake_neuralset(monkeypatch)
    sys.modules.pop("brain2qwerty_v1.utils", None)
    return importlib.import_module("brain2qwerty_v1.utils")


def _segment(uid):
    return SimpleNamespace(trigger=SimpleNamespace(extra={"sentence_UID": uid}))


def test_sampler_keeps_sentence_grouping_on_single_rank(monkeypatch):
    utils = _load_utils(monkeypatch)
    sampler = utils.SentenceGroupedDistributedSampler(
        [_segment("A"), _segment("B"), _segment("A"), _segment("C")],
        shuffle=False,
    )
    assert list(iter(sampler)) == [0, 2, 1, 3]
    assert len(sampler) == 4


def test_sampler_pads_each_rank_to_equal_length(monkeypatch):
    utils = _load_utils(monkeypatch)
    sampler = utils.SentenceGroupedDistributedSampler(
        [_segment("A"), _segment("B"), _segment("A"), _segment("C")],
        shuffle=False,
    )

    sampler._rank_world = lambda: (0, 2)
    assert list(iter(sampler)) == [0, 2, 3]
    assert len(sampler) == 3

    sampler._rank_world = lambda: (1, 2)
    assert list(iter(sampler)) == [1, 1, 1]


def test_sampler_shuffle_is_seeded_and_epoch_dependent(monkeypatch):
    utils = _load_utils(monkeypatch)
    sampler = utils.SentenceGroupedDistributedSampler(
        [_segment("A"), _segment("B"), _segment("C"), _segment("D")],
        seed=7,
        shuffle=True,
    )

    sampler.set_epoch(0)
    order_epoch0_a = list(iter(sampler))
    order_epoch0_b = list(iter(sampler))
    assert order_epoch0_a == order_epoch0_b

    # Adjacent RNG seeds can occasionally produce the same permutation for tiny
    # sets, so assert epoch-dependence by requiring more than one unique order
    # across several epochs.
    orders = set()
    for epoch in range(6):
        sampler.set_epoch(epoch)
        orders.add(tuple(iter(sampler)))
    assert len(orders) > 1
