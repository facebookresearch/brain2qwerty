# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Grid search launcher: train on all subjects with multiple splitting seeds.

python -m brain2qwerty.grids.run_grid
"""

from neuralset.infra import ConfDict
from neuraltrain.utils import run_grid

from ..main import Experiment
from .defaults import SAVEDIR, default_config

GRID_NAME = "brain2qwerty_grid"

update = {
    "infra": {
        "cluster": "auto",
        "folder": SAVEDIR,
        "timeout_min": 48 * 60,
        "gpus_per_node": 1,
        "cpus_per_task": 10,
        "slurm_constraint": "volta32gb",
        "job_name": GRID_NAME,
    },
    "data.study.query": None,
    "n_epochs": 100,
}

grid = {
    "data.splitting_seed": [0, 1, 2],
}

if __name__ == "__main__":
    updated_config = ConfDict(default_config)
    updated_config.update(update)

    run_grid(
        Experiment,
        GRID_NAME,
        updated_config,
        grid,
        job_name_keys=["infra.wandb_config.name", "infra.job_name"],
        combinatorial=True,
        overwrite=True,
        dry_run=False,
    )
