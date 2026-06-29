import argparse
from types import SimpleNamespace

from brain2qwerty_v1.cli import add_wandb_args as add_v1_args
from brain2qwerty_v1.cli import wandb_config as v1_wandb_config
from brain2qwerty_v2.cli import add_wandb_args as add_v2_args
from brain2qwerty_v2.cli import wandb_config as v2_wandb_config


def test_v1_add_wandb_args_defaults():
    parser = argparse.ArgumentParser()
    add_v1_args(parser)
    args = parser.parse_args([])
    assert args.wandb is False
    assert args.wandb_project == "brain2qwerty_v1"


def test_v2_add_wandb_args_defaults():
    parser = argparse.ArgumentParser()
    add_v2_args(parser)
    args = parser.parse_args([])
    assert args.wandb is False
    assert args.wandb_project == "brain2qwerty_v2"


def test_v1_wandb_config_disabled_returns_none():
    args = SimpleNamespace(wandb=False)
    assert v1_wandb_config(args, command="train", seed=7) is None


def test_v2_wandb_config_enabled_uses_command_group_fallback(monkeypatch):
    monkeypatch.setenv("WANDB_HOST", "http://wandb.local")
    args = SimpleNamespace(
        wandb=True,
        wandb_project="proj",
        wandb_group=None,
        wandb_entity="ent",
    )
    out = v2_wandb_config(args, command="eval", seed=3)
    assert out == {
        "project": "proj",
        "group": "eval",
        "entity": "ent",
        "name": "eval-seed3",
        "host": "http://wandb.local",
    }
