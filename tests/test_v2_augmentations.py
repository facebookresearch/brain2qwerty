import torch

from brain2qwerty_v2.augmentations import Preprocess, PreprocessConfig


def test_preprocess_config_forbids_unknown_keys():
    cfg = PreprocessConfig(whiteNoiseSD=0.1)
    assert cfg.whiteNoiseSD == 0.1


def test_preprocess_noop_keeps_shape_and_sizes():
    p = Preprocess(whiteNoiseSD=0.0, constantOffsetSD=0.0)
    batch = {
        "neuros": torch.ones(2, 5, 3),
        "neuro_sizes": torch.tensor([5, 5], dtype=torch.long),
    }
    out = p(batch)
    assert out["neuros"].shape == (2, 5, 3)
    assert torch.equal(out["neuro_sizes"], torch.tensor([5, 5]))


def test_preprocess_time_stretch_updates_lengths(monkeypatch):
    p = Preprocess(whiteNoiseSD=0.0, constantOffsetSD=0.0, time_stretch=True)

    class _StretchFactor:
        def uniform_(self, _a, _b):
            return self

        def item(self):
            return 1.2

    monkeypatch.setattr("brain2qwerty_v2.augmentations.torch.empty", lambda *_a, **_k: _StretchFactor())

    batch = {
        "neuros": torch.randn(1, 10, 4),
        "neuro_sizes": torch.tensor([10], dtype=torch.long),
    }
    out = p(batch)
    assert out["neuros"].shape[1] == 12
    assert out["neuro_sizes"].item() == 12
