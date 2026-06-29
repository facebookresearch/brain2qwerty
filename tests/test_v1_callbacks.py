import json
from pathlib import Path
from types import SimpleNamespace

import torch

from brain2qwerty_v1.callbacks import LogSentencePredictions


class _Trainer:
    def __init__(self, save_dir, rank=0):
        self.global_rank = rank
        self.logger = None
        self._save_dir = save_dir


class _Segment:
    def __init__(self, uid, sentence):
        self.trigger = SimpleNamespace(extra={"sentence_UID": uid}, sentence=sentence)


def test_log_sentence_predictions_writes_test_json(tmp_path):
    callback = LogSentencePredictions(save_dir=str(tmp_path))
    trainer = _Trainer(save_dir=tmp_path, rank=0)

    callback.on_test_epoch_start(trainer, pl_module=None)

    y_pred = torch.tensor([[0.1, 0.9, 0.0], [0.9, 0.1, 0.0]])
    y_true = torch.tensor([1, 0])
    batch = SimpleNamespace(
        segments=[_Segment("sent-1", "hola"), _Segment("sent-1", "hola")]
    )

    callback.on_test_batch_end(
        trainer,
        pl_module=None,
        outputs=(y_pred, y_true),
        batch=batch,
        batch_idx=0,
    )
    callback.on_test_epoch_end(trainer, pl_module=None)

    out_file = Path(tmp_path) / "callbacks" / "test_all_sentences.json"
    assert out_file.exists()

    payload = json.loads(out_file.read_text())
    assert set(payload.keys()) == {"sent-1"}
    assert payload["sent-1"]["true"] == "hola"
    assert payload["sent-1"]["typed"] == [1, 0]
    assert payload["sent-1"]["pred"] == [1, 0]
    assert len(payload["sent-1"]["logits"]) == 2


def test_log_sentence_predictions_skips_non_zero_rank(tmp_path):
    callback = LogSentencePredictions(save_dir=str(tmp_path))
    trainer = _Trainer(save_dir=tmp_path, rank=1)
    callback.on_test_epoch_start(trainer, pl_module=None)

    callback.on_test_batch_end(
        trainer,
        pl_module=None,
        outputs=(torch.tensor([[1.0, 0.0]]), torch.tensor([0])),
        batch=SimpleNamespace(segments=[_Segment("s", "x")]),
        batch_idx=0,
    )
    callback.on_test_epoch_end(trainer, pl_module=None)

    assert not (Path(tmp_path) / "callbacks" / "test_all_sentences.json").exists()
