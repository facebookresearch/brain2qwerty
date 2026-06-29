import sys
import types


class _DataStub:
    def __init__(self, sink):
        self._sink = sink

    def build(self):
        self._sink["built"] = self._sink.get("built", 0) + 1


class _ExpStub:
    calls = []

    def __init__(self, **cfg):
        self.cfg = cfg
        self.data = _DataStub(cfg)
        _ExpStub.calls.append(cfg)

    def run(self):
        self.cfg["ran"] = True


def test_v1_main_cache_and_eval_paths(monkeypatch):
    import brain2qwerty_v1.main as m

    _ExpStub.calls = []
    monkeypatch.setitem(sys.modules, "studies", types.ModuleType("studies"))
    monkeypatch.setattr(m, "Experiment", _ExpStub)
    monkeypatch.setattr("brain2qwerty_v1.config.xp_config.debug_config", lambda: {"seed": 1})
    monkeypatch.setattr(
        "brain2qwerty_v1.config.xp_config.experiment_config", lambda: {"seed": 2}
    )
    monkeypatch.setattr(
        "brain2qwerty_v1.cli.wandb_config", lambda args, command, seed: {"wb": True}
    )

    m.main(["cache", "--debug"])
    assert _ExpStub.calls[-1]["seed"] == 1
    assert _ExpStub.calls[-1]["built"] == 1

    m.main(["eval", "--ckpt", "model.ckpt", "--wandb"])
    cfg = _ExpStub.calls[-1]
    assert cfg["eval_only"] is True
    assert cfg["ckpt_path"] == "model.ckpt"
    assert cfg["wandb_config"] == {"wb": True}
    assert cfg["ran"] is True


def test_v1_main_train_overrides_seed(monkeypatch):
    import brain2qwerty_v1.main as m

    _ExpStub.calls = []
    monkeypatch.setitem(sys.modules, "studies", types.ModuleType("studies"))
    monkeypatch.setattr(m, "Experiment", _ExpStub)
    monkeypatch.setattr(
        "brain2qwerty_v1.config.xp_config.experiment_config", lambda: {"seed": 11}
    )
    monkeypatch.setattr("brain2qwerty_v1.cli.wandb_config", lambda *a, **k: None)

    m.main(["train", "--seed", "99"])
    cfg = _ExpStub.calls[-1]
    assert cfg["seed"] == 99
    assert cfg["ran"] is True


def test_v2_main_cache_eval_and_resume(monkeypatch):
    import brain2qwerty_v2.main as m

    _ExpStub.calls = []
    monkeypatch.setitem(sys.modules, "studies", types.ModuleType("studies"))
    monkeypatch.setattr(m, "Experiment", _ExpStub)
    monkeypatch.setattr("brain2qwerty_v2.config.xp_config.debug_config", lambda: {"seed": 3})
    monkeypatch.setattr(
        "brain2qwerty_v2.config.xp_config.experiment_config", lambda: {"seed": 4}
    )
    monkeypatch.setattr(
        "brain2qwerty_v2.cli.wandb_config", lambda args, command, seed: {"wb2": True}
    )

    m.main(["cache", "--debug"])
    assert _ExpStub.calls[-1]["seed"] == 3
    assert _ExpStub.calls[-1]["built"] == 1

    m.main(["eval", "--ckpt", "best.ckpt", "--wandb"])
    cfg_eval = _ExpStub.calls[-1]
    assert cfg_eval["eval_only"] is True
    assert cfg_eval["ckpt_path"] == "best.ckpt"
    assert cfg_eval["wandb_config"] == {"wb2": True}
    assert cfg_eval["ran"] is True

    m.main(["train", "--resume", "resume.ckpt", "--seed", "7"])
    cfg_train = _ExpStub.calls[-1]
    assert cfg_train["resume_ckpt"] == "resume.ckpt"
    assert cfg_train["seed"] == 7
    assert cfg_train["ran"] is True
