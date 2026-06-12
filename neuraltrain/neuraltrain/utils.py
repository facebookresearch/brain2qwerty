# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility scripts."""

from __future__ import annotations

import inspect
import pickle
import shutil
import typing as tp
from itertools import product
from pathlib import Path

import pydantic
import submitit
import wandb
from pydantic import BaseModel, Field, create_model

from neuralset.infra import ConfDict, TaskInfra  # type: ignore[import]


def convert_to_pydantic(
    class_to_convert: type,
    name: str,
    parent_class: tp.Any = None,
    exclude_from_build: list[str] | None = None,
) -> BaseModel:
    """
    Converts any class into a pydantic BaseModel. Initialize the class
    with the 'self.build()' method
    """
    # Get the constructor of the class
    init = class_to_convert.__init__  # type: ignore

    # Inspect signature
    sig = inspect.signature(init)
    empty = inspect.Parameter.empty

    fields = {
        k: (
            v.annotation if v.annotation != empty else tp.Any,
            v.default if v.default != empty else ...,
        )
        for k, v in sig.parameters.items()
        if k != "self" and not k.startswith("_")
    }

    # add name for pydantic.discriminator
    assert "name" not in sig.parameters.items()

    # Create the Pydantic model class dynamically
    Builder = create_model(  # type: ignore
        name,
        name=(tp.Literal[name], Field(default=name)),
        __base__=parent_class,
        **fields,
    )
    Builder._cls = class_to_convert  # type: ignore

    # Define a build method to instantiate the original class
    if exclude_from_build is None:
        exclude_from_build = []

    def build_method(instance: BaseModel):
        params = dict(
            (field, getattr(instance, field))
            for field in type(instance).model_fields
            if (field != "name" and field not in exclude_from_build)
        )
        return instance._cls(**params)  # type: ignore

    # Bind the build method to Builder instances using MethodType
    setattr(Builder, "build", build_method)

    return Builder


def all_subclasses(cls):
    """Get all subclasses of cls recursively."""
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)]
    )


class BaseExperiment(pydantic.BaseModel):
    """Base experiment class which require an infra and a 'run' method."""

    infra: TaskInfra = TaskInfra()

    @classmethod
    def _exclude_from_cls_uid(cls) -> tp.List[str]:
        return []

    def run(self):
        raise NotImplementedError


def run_grid(
    exp_cls: tp.Type[BaseExperiment],
    exp_name: str,
    base_config: dict[str, tp.Any],
    grid: dict[str, list],
    job_name_keys: list[str] | None = None,
    combinatorial: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    infra_mode: str = "retry",
) -> list[ConfDict]:
    """Run grid over provided experiment.

    Parameters
    ----------
    exp_cls :
        Experiment class to instantiate with `grid`. Must have an `infra` attribute, which will be
        updated when instantiating the different experiments of the grid.
    exp_name :
        Name of the base experiment to run.
    grid :
        Dictionary containing values to perform the sweep on.
    base_config :
        Base configuration to update.
    job_name_keys :
       Flattened config key(s) to update with the experiment-specific 'job_name' variable. E.g.,
       can be used to pass the job name to a wandb logger.
    combinatorial :
        If True, run grid over all possible combinations of the grid. If False, run each parameter
        change individually.
    overwrite :
        If True, delete existing experiment-specific folder.
    dry_run :
        If True, do not add tasks to the infra.
    infra_mode :
        Whether to rerun existing or failed experiments.
        - cached: cache is returned if available (error or not),
                otherwise computed (and cached)
        - retry: cache is returned if available except if it's an error,
                otherwise (re)computed (and cached)
        - force: cache is ignored, and result is (re)computed (and cached)

    Returns
    -------
    list :
        List of config dictionaries used for each experiment of the grid.
    """
    job_array_kwargs = {}
    if dry_run:
        from importlib.metadata import version

        from pkg_resources import parse_version

        if parse_version(version("exca")) < parse_version("0.4.5"):
            raise ImportError("`dry_run` requires `exca>=0.4.5` to be installed.")
        job_array_kwargs["allow_empty"] = True

    # Update savedir of experiment infra
    base_config = base_config
    base_folder = Path(base_config["infra"]["folder"])

    task: BaseExperiment = exp_cls(
        **base_config,
    )

    if combinatorial:
        grid_product = list(dict(zip(grid.keys(), v)) for v in product(*grid.values()))
    else:
        grid_product = [
            {param: value} for param, values in grid.items() for value in values
        ]

    print(f"Launching {len(grid_product)} tasks")

    out_configs = []
    tmp = task.infra.clone_obj(**{"infra.mode": infra_mode})
    with tmp.infra.job_array(**job_array_kwargs) as tasks:
        for params in grid_product:
            job_name = ConfDict(params).to_uid()

            config = ConfDict(base_config)
            config.update(params)

            folder = base_folder / exp_name / job_name
            if folder.exists():  # FIXME: adapt to checkpointing
                print(f"{folder} already exists.")
                if overwrite and not dry_run:
                    print(f"Deleting {folder}.")
                    shutil.rmtree(folder)
                    folder.mkdir()

            # Update infra and logger
            config["infra.folder"] = str(folder)
            if job_name_keys is not None:
                for key in job_name_keys:
                    config.update({key: str(job_name)})

            if not dry_run:
                task_ = exp_cls(**config)
                tasks.append(task_)

            out_configs.append(config)

    print("Done.")

    return out_configs


class WandbLoggerConfig(pydantic.BaseModel):
    """
    Pydantic configuration for torch-lightning's wandb logger.
    See https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html.
    If you want to resume a run, you can use the `id` field to specify the run id, either in the config or in the `build` method.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    # core fields
    name: str | None = None
    group: str
    entity: str | None = None
    project: str | None = None
    # extra fields
    offline: bool = False
    host: str | None = None
    id: str | None = None
    dir: Path | None = None
    anonymous: bool | None = None
    log_model: str | bool = False
    experiment: tp.Any | None = None
    prefix: str = ""
    resume: tp.Literal["allow", "never", "must"] = "allow"

    # pylint: disable=redefined-builtin
    def build(
        self,
        save_dir: str | Path,
        xp_config: dict | pydantic.BaseModel | None = None,
        id: str | None = None,
    ) -> tp.Any:
        if self.offline:
            login_kwargs = {"key": "X" * 40}
        else:
            login_kwargs = {"host": self.host}  # type: ignore
        wandb.login(**login_kwargs)  # type: ignore
        from lightning.pytorch.loggers import WandbLogger

        if isinstance(xp_config, pydantic.BaseModel):
            xp_config = xp_config.model_dump()
        config = self.model_dump()
        if id is not None:
            config["id"] = id
        del config["host"]
        logger = WandbLogger(**config, save_dir=save_dir, config=xp_config)
        try:
            logger.experiment.config["_dummy"] = None  # To launch initialization
        except TypeError:
            pass  # Crashes if called in a second process, e.g. with DDP
        return logger


class WandbInfra(TaskInfra):
    wandb_config: WandbLoggerConfig | None = None

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.wandb_config and self.wandb_config.group is not None:
            self.version = self.wandb_config.group

    def _wandb_uid(self):
        if self.wandb_config.group is not None and self.wandb_config.name is not None:
            uid = self.wandb_config.group + "-" + self.wandb_config.name
            uid = uid[:128]
        else:
            uid = self.uid().split("-")[-1]
        for bad_char in "/:,=[]{}()":
            uid = uid.replace(bad_char, ".")
        return uid

    def _run_method(self, *args, **kwargs):
        out = super()._run_method(*args, **kwargs)

        if self.wandb_config is not None:
            try:
                with wandb.init(
                    project=self.wandb_config.project,
                    entity=self.wandb_config.entity,
                    group=self.wandb_config.group,
                    name=self.wandb_config.name,
                ) as run:
                    artifact = wandb.Artifact(self._wandb_uid(), type="pkl")
                    wandb_folder = self.uid_folder() / "wandb"
                    wandb_folder.mkdir(exist_ok=True, parents=True)
                    with open(wandb_folder / "output.pkl", "wb") as f:
                        pickle.dump(out, f)
                    self.config(uid=False, exclude_defaults=False).to_yaml(
                        wandb_folder / "config.yaml"
                    )
                    fnames = [wandb_folder / "config.yaml", wandb_folder / "output.pkl"]
                    try:
                        env = submitit.JobEnvironment()
                        fnames += [env.paths.stderr, env.paths.stdout]
                    except:
                        pass  # Not running in submitit
                    for fname in fnames:
                        artifact.add_file(fname)
                    run.log_artifact(artifact)
                    print(f"Uploaded to wandb: {self._wandb_uid()}")
                    (wandb_folder / "output.pkl").unlink()
            except wandb.errors.CommError:
                print("Could not connect to wandb. Skipping upload")
        return out

    def download(self, version="v0") -> tp.Any:
        if self.uid_folder().exists():  # type: ignore
            print(f"Folder {self.uid_folder()} already exists.")
            return
        if self.wandb_config is None:
            raise ValueError(
                "wandb_config must be provided to download artifacts from wandb."
            )
        with wandb.init(
            project=self.wandb_config.project, entity=self.wandb_config.entity
        ) as run:
            artifact = run.use_artifact(f"{self._wandb_uid()}:{version}")
            artifact.download(self.uid_folder())
        return artifact
