import csv
from pathlib import Path
from types import SimpleNamespace

from brain2qwerty_v2.callbacks import PredictionCSVCallback


def test_prediction_csv_callback_writes_rows(monkeypatch, tmp_path):
    callback = PredictionCSVCallback(save_dir=str(tmp_path))
    trainer = SimpleNamespace(world_size=1, global_rank=0)
    module = SimpleNamespace(
        _test_predictions=[
            {
                "true_text": "hola",
                "pred_text": "hola",
                "ctc_text": "hola",
                "subject": "S1",
                "sentence_UID": "uid-1",
            }
        ]
    )

    monkeypatch.setattr(
        "brain2qwerty_v2.callbacks.compute_sample_metrics",
        lambda true_texts, pred_texts, ctc_texts=None, with_semer=True: [
            {
                "true_text": true_texts[0],
                "pred_text": pred_texts[0],
                "ctc_text": ctc_texts[0],
                "CER": 0.0,
                "CTC_CER": 0.0,
                "WER": 0.0,
                "SemER": 0.0,
            }
        ],
    )

    callback.on_test_epoch_end(trainer, module)

    out = Path(tmp_path) / "predictions_test.csv"
    assert out.exists()
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["true_text"] == "hola"
    assert rows[0]["subject"] == "S1"
    assert rows[0]["sentence_UID"] == "uid-1"


def test_prediction_csv_callback_skips_nonzero_rank(tmp_path):
    callback = PredictionCSVCallback(save_dir=str(tmp_path))
    trainer = SimpleNamespace(world_size=1, global_rank=1)
    rows = [{"true_text": "a", "pred_text": "a"}]

    callback._save(trainer, rows, "predictions_test.csv", with_semer=False)

    assert not (Path(tmp_path) / "predictions_test.csv").exists()
