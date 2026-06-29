import torch

from brain2qwerty_v2.metrics import CharacterErrorRate, SemanticErrorRate


def test_character_error_rate_perfect_prediction_is_zero():
    metric = CharacterErrorRate()
    y_pred = torch.zeros(1, 4, 5)
    y_pred[0, 0, 1] = 1.0
    y_pred[0, 1, 2] = 1.0
    y_pred[0, 2, 3] = 1.0
    y_pred[0, 3, 0] = 1.0  # blank

    y_true = torch.tensor([[1, 2, 3]], dtype=torch.long)
    adjusted_x_len = torch.tensor([4], dtype=torch.long)
    y_len = torch.tensor([3], dtype=torch.long)

    metric.update(y_pred, y_true, adjusted_x_len, y_len)
    assert float(metric.compute()) == 0.0


def test_character_error_rate_empty_target_returns_zero():
    metric = CharacterErrorRate()
    assert float(metric.compute()) == 0.0


def test_semantic_error_rate_with_mocked_encoder(monkeypatch):
    metric = SemanticErrorRate(batch_size=2)

    def _fake_encode(sentences):
        return torch.tensor([[1.0, 0.0], [0.0, 1.0]])[: len(sentences)]

    monkeypatch.setattr(metric, "_encode", _fake_encode)
    metric.update(["a", "b"], ["a", "b"])
    assert float(metric.compute()) == 0.0
