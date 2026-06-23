# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared argument helpers for the command entry points (train / eval / debug)."""

import argparse
import os

PROJECT = "brain2qwerty_v2"


def add_wandb_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    parser.add_argument("--wandb-project", default=PROJECT)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-entity", default=None)


def wandb_config(args, command: str, seed) -> dict | None:
    if not getattr(args, "wandb", False):
        return None
    return {
        "project": args.wandb_project,
        "group": args.wandb_group or command,
        "entity": args.wandb_entity,
        "name": f"{command}-seed{seed}",
        "host": os.environ.get("WANDB_HOST"),
    }
