# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import logging
import typing as tp
import warnings
from pathlib import Path

import numpy as np
import pydantic
import yaml
from typing_extensions import Annotated

PathLike = str | Path


# # # # # CONFIGURE LOGGER # # # # #
logger = logging.getLogger("neuralset")
_handler = logging.StreamHandler()
_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s", "%Y-%m-%d %H:%M:%S"
)
_handler.setFormatter(_formatter)
logger.addHandler(_handler)
logger.setLevel(logging.INFO)
# # # # # CONFIGURED LOGGER # # # # #


def _int_cast(v: tp.Any) -> tp.Any:
    """casts integers to string"""
    if isinstance(v, int):
        return str(v)
    return v


# type hint for casting integers to string
# this is useful for subject field which can be automatically converted from
# str to int by pandas
StrCast = Annotated[str, pydantic.BeforeValidator(_int_cast)]
CACHE_FOLDER = Path.home() / ".cache/neuralset/"
CACHE_FOLDER.mkdir(parents=True, exist_ok=True)


class _Module(pydantic.BaseModel):
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ()
    model_config = pydantic.ConfigDict(protected_namespaces=(), extra="forbid")

    @classmethod
    def _exclude_from_cls_uid(cls) -> tp.List[str]:
        return []

    @tp.final  # make sure nobody gets it wrong and override it
    def __post_init__(self) -> None:
        """This should not exist in subclasses, as we use pydantic's model_post_init"""

    @classmethod
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        # get requirements from superclasses as well
        reqs = tuple(x.strip() for x in cls.requirements)
        for base in cls.__bases__:
            breqs = getattr(base, "requirements", ())
            if breqs is not cls.requirements:
                reqs = breqs + reqs
        cls.requirements = reqs

    @classmethod
    def _can_be_instanciated(cls) -> bool:
        return not any(cls.__name__.startswith(k) for k in ["Base", "_"])

    @classmethod
    def install_requirements(cls) -> None:
        cls._check_requirements(install=True)

    @classmethod
    def _check_requirements(cls, install: bool = False) -> None:
        import_names = {
            "pillow": "PIL",
            "scikit-image": "skimage",
            "opencv-python": "cv2",
            "git+https://github.com/nltk/nltk_contrib.git@683961c53f0c122b90fe2d039fe795e0a2b3e997": "nltk_contrib",
        }

        for package in cls.requirements:
            name = package.split(">=")[0]
            name = name.split("==")[0]
            try:
                importlib.import_module(import_names.get(name, name))
            except ModuleNotFoundError:
                if install:
                    # importing pip has border effects on distutils (and make a mess with dino)
                    import pip

                    warnings.warn(f"Installing missing package {name!r} (this may crash)")
                    pip.main(["install", package])
                else:
                    warnings.warn(
                        f"Missing {name!r}. This will "
                        f"likely crash. Use {cls.__name__}"
                        ".install_requirements()"
                    )


class Frequency(float):
    """A float representing a frequency, with extra helpers to
    help convert from seconds to samples and vice-versa
    """

    @tp.overload
    def to_ind(self, seconds: float) -> int: ...

    @tp.overload  # noqa
    def to_ind(self, seconds: np.ndarray) -> np.ndarray:  # noqa
        ...

    def to_ind(self, seconds: tp.Any) -> tp.Any:  # noqa
        """Converts a time in seconds (or multiple times in an array)
        to a sample index
        """
        if isinstance(seconds, np.ndarray):
            return np.round(seconds * self).astype(int)
        return int(round(seconds * self))

    @tp.overload
    def to_sec(self, index: int) -> float: ...

    @tp.overload  # noqa
    def to_sec(self, index: np.ndarray) -> np.ndarray:  # noqa
        ...

    def to_sec(self, index: tp.Any) -> tp.Any:  # noqa
        """Converts a sample index to a time in seconds"""
        return index / self

    @staticmethod
    def _yaml_representer(dumper, data):
        "Represents Frequency instances as floats in yamls"
        return dumper.represent_scalar("tag:yaml.org,2002:float", str(float(data)))


class TimedArray:
    def __init__(
        self,
        *,  # forbid positional
        frequency: float,
        start: float,
        data: np.ndarray | None = None,
        duration: float | None = None,
        aggregation: str = "sum",  # sum or average
    ) -> None:
        """Numpy array with time and frequency attached to it.
        This aims at facilitating slice creations.
        The time dimension must be the last dimension.

        Parameters
        ----------
        frequency: float
            sampling frequency of the data. If >0, then the last
            dimension of the data should be the time dimension, and if 0
            the data should not have any time dimension.
        start: float
            start time of the data
        data: optional array
            if provided, the data with time as last dimension if frequency>0
        duration: optional float
            if provided, the duration of the data. If data is also provided
            and frequency>0, last dimension will be checked for consistency
            If data is not provided, the TimedArray will get its shape from
            the first data added to it.
        aggregation: "sum" or "average"
            aggregation mode on the time domain when adding to the timed array
        """
        self.frequency = Frequency(frequency)
        self.start = start
        self.aggregation = aggregation
        exp_size = 0
        if duration is not None and duration < 0:
            raise ValueError(f"duration should be None or >=0, got {duration}")

        if data is None:
            if duration is None:
                raise ValueError("Missing data or duration")
            # post-poned initialization of data through __iadd__
            # initialize with data.size == 0
            if not frequency:
                data = np.zeros((0,))
            else:
                exp_size = max(1, self.frequency.to_ind(duration))
                data = np.zeros((0, exp_size))
        self.data = data
        if frequency and duration is not None:
            exp_size = 0 if not duration else max(1, self.frequency.to_ind(duration))
            if duration and not self.data.shape[-1]:
                msg = "Last dimension is empty but frequency and duration are not null "
                msg += f"(shape={self.data.shape})"
                raise ValueError(msg)
            if abs(data.shape[-1] - exp_size) > 1:
                msg = f"Data has incorrect (last) dimension {data.shape} for duration "
                msg += f"{duration} and frequency {frequency} (expected {exp_size})"
                raise ValueError(msg)
        if frequency:
            self.duration = self.frequency.to_sec(data.shape[-1])
        elif duration is None:
            raise ValueError(f"duration must be provided if {frequency=}")
        else:
            self.duration = duration
        # averaging
        self._overlapping_data_count: None | np.ndarray = None
        if aggregation == "average":
            num = self.data.shape[-1] if self.frequency else 1
            self._overlapping_data_count = np.zeros(num, dtype=int)
        elif aggregation != "sum":
            raise ValueError(f"Unknown {aggregation=}")

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        fields = "frequency,start,duration,aggregation,data".split(",")
        string = ",".join(f"{f}={getattr(self, f)}" for f in fields)
        return f"{cls}({string})"

    def __iadd__(self, other: "TimedArray") -> "TimedArray":
        if other.frequency and self.frequency != other.frequency:
            diff = abs(self.frequency - other.frequency)
            if diff * max(self.duration, other.duration) >= 0.5:  # half sample diff
                msg = f"Cannot add with different (non-0) frequencies ({other.frequency} and {self.frequency})"
                raise ValueError(msg)
        if not self.data.size:
            # post-poned initialization of data, recover shape from other.data
            last = -1 if other.frequency else None
            shape = other.data.shape[:last]
            if self.frequency:
                shape += (self.data.shape[-1],)
            self.data = np.zeros(shape, dtype=other.data.dtype)
        if self.frequency:
            slices = [
                sa1._overlap_slice(sa2.start, sa2.duration)
                for sa1, sa2 in [(self, other), (other, self)]
            ]
            if slices[0] is None or slices[1] is None:
                return self
            # slices
            self_slice = slices[0][-1]
            other_slice = slices[1][-1]
        else:
            self_slice = None
            other_slice = None
        if self._overlapping_data_count is None:  # sum
            self.data[..., self_slice] += other.data[..., other_slice]
        else:  # average
            counts = self._overlapping_data_count[..., self_slice]
            upd = counts / (1.0 + counts)
            self.data[..., self_slice] *= upd
            self.data[..., self_slice] += (1 - upd) * other.data[..., other_slice]
            counts += 1
        return self

    def _overlap_slice(
        self, start: float, duration: float
    ) -> tuple[float, float, slice | None] | None:
        if duration < 0:
            raise ValueError(f"duration should be >=0, got {duration=}")
        overlap_start = max(start, self.start)
        overlap_stop = min(start + duration, self.start + self.duration)
        if overlap_stop < overlap_start:
            return None  # no overlap
        if overlap_stop == overlap_start and self.duration and duration:
            return None  # 2 timed arrays with durations with one starting when the other ends
        if not self.frequency:
            return overlap_start, overlap_stop - overlap_start, None
        if not self.duration:
            return None  # frequency but no duration -> empty
        start_ind = self.frequency.to_ind(overlap_start - self.start)
        duration_ind = self.frequency.to_ind(overlap_stop - overlap_start)
        # # # right edge border case # # #
        if duration_ind <= 0:  # faster than max
            duration_ind = 1
        # then make sure we move the start according to the number of selected samples
        tps = self.data.shape[-1]
        if start_ind > tps - duration_ind:
            start_ind = tps - duration_ind
        if start_ind < 0:
            raise RuntimeError(f"Fail for {start=} {duration=} on {self}")
        start = self.frequency.to_sec(start_ind) + self.start
        duration = self.frequency.to_sec(duration_ind)
        # # # build # # #
        out = start, duration, slice(start_ind, start_ind + duration_ind)
        return out

    def overlap(self, start: float, duration: float) -> "TimedArray":
        """Returns the sub TimedArray overlapping with the provided start
        and duration
        In case of lack of overlap, a timed array with 0 duration and empty
        data on the time dimension will be returned.
        """
        out = self._overlap_slice(start, duration)
        if out is not None:
            ostart, oduration, sl = out
        else:
            ostart, oduration, sl = min(start, self.start), 0, slice(0, 0)
        return TimedArray(
            frequency=self.frequency,
            start=ostart,
            duration=oduration,
            data=self.data[..., sl],
        )


yaml.representer.SafeRepresenter.add_representer(Frequency, Frequency._yaml_representer)
