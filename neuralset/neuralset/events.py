# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Event handling classes and functions.

The class `Event` and its children (e.g. `Word`, `Meg`, etc.) define the
expected fields for each event type.
"""

import functools
import inspect
import logging
import typing as tp
import urllib
from abc import abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import pydantic

from .base import Frequency, StrCast, _Module
from .utils import ignore_all, warn_once

E = tp.TypeVar("E", bound="Event")
logger = logging.getLogger(__name__)


class Event(_Module):
    """Base class for all event types with the bare minimum common fields.

    If the event is instantiated with `from_dict()`, additional non-required
    fields that are provided will be ignored instead of causing an error.
    """

    start: float
    timeline: str
    duration: pydantic.NonNegativeFloat = 0.0
    extra: dict[str, tp.Any] = {}
    type: tp.ClassVar[str] = "Event"
    _CLASSES: tp.ClassVar[dict[str, tp.Type["Event"]]] = {}
    _index: int | None = None  # records index in dataframe for debugging

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        # register all events
        cls.type = cls.__name__
        Event._CLASSES[cls.__name__] = cls

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if pd.isna(self.start):
            raise ValueError(f"Start time needs to be provided for {self!r}")

    @classmethod
    def from_dict(cls: tp.Type[E], row: tp.Any) -> E:
        """Create event from dictionary/row/named-tuple while grouping non-specified fields
        under "extra" dict attribute

        Legacy/deprecated: "extra__" is removed from keys in "extra" dict attribute
        """
        index: int | None = None
        if hasattr(row, "_asdict"):
            index = getattr(row, "Index", None)
            row = row._asdict()  # supports named tuples
        cls_ = cls._CLASSES[row["type"]]
        if not issubclass(cls_, cls):
            raise TypeError(f"{cls_} is not a subclass of {cls}")
        fs = set(cls_.model_fields)  # type: ignore
        kwargs: dict[str, tp.Any] = {}
        extra = {}
        for k, v in row.items():
            if pd.isna(v):  # all nans are ignored
                continue
            if k in fs:
                kwargs[k] = v
            elif k != "type":
                if k.startswith("extra__"):  # legacy
                    k = k[7:]
                extra[k] = v
        kwargs.setdefault("extra", {}).update(extra)  # can bug if extra is a column
        try:
            out = cls_(**kwargs)
        except Exception as e:
            logger.warning(
                "Event.from_dict parsing failed for input %s\nmapped to %s\n with error: %s)",
                row.to_string() if hasattr(row, "to_string") else row,
                kwargs,
                e,
            )
            raise
        out._index = index
        return out

    def to_dict(self) -> dict[str, tp.Any]:
        """Export the event as a dictionary usable for csv dump.
        "extra" dictionary field is flatened into "extra.subfield" fields to simplify
        queries through pandas.
        """
        out = dict(self.extra)
        out["type"] = self.type
        # avoid Path in exports
        tag = "extra"
        fields = {x: str(y) if isinstance(y, Path) else y for x, y in self if x != tag}
        out.update(fields)
        return out

    @property
    def stop(self) -> float:
        return self.start + self.duration

    def __str__(self) -> str:
        core_fields = {k: v for k, v in self if k != "extra"}
        return ", ".join([f"{k}={v}" for k, v in core_fields.items()])


Event._CLASSES["Event"] = Event


class EventTypesHelper:
    """Computes and stores information about the event types
    provided either as an actual type, or a type name, or a tuple of type names
    to get a unified and simple access while the event type can be specified
    in multiple ways.

    Parameter
    ---------
    event_types: Event type, or str, or tuple of str
        event type or name of an event or tuple of names of events

    Attributes
    ----------
    classes: tuple of Event types
        the classes specified as event types (as a tuple even if only 1 type was specified)
    names: str
        the list of event type names specified, including subclasses. This is particularly
        handy to filter a dataframe: :code:`events[events.type.isin(helper.names)]`
    """

    def __init__(self, event_types: str | tp.Type[Event] | tp.Sequence[str]) -> None:
        self.specified = event_types
        if inspect.isclass(event_types):
            self.classes: tp.Tuple[tp.Type[Event], ...] = (event_types,)
        else:
            if isinstance(event_types, str):
                event_types = (event_types,)
            try:
                self.classes = tuple(Event._CLASSES[x] for x in event_types)  # type: ignore
            except KeyError as e:
                avail = list(Event._CLASSES)
                msg = f"{event_types} is an invalid event name, use one of {avail}"
                raise ValueError(msg) from e
        items = Event._CLASSES.items()
        self.names = [x for x, y in items if issubclass(y, self.classes)]


class BaseDataEvent(Event):
    """A base class for events who's data needs to be read from a file."""

    filepath: Path | str = ""
    frequency: float = 0
    _read_method: tp.Any = None

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if not self.filepath:
            raise ValueError("A filepath must be provided")
        # check whether file actually points to register
        self._set_read_method()
        fp = str(self.filepath)
        self.filepath = fp
        if ":" not in str(fp):  # deactivate check
            # make sure to store file as a string in dataframe
            if not Path(fp).exists():
                warn_once(f"file missing: {fp}")

    def _set_read_method(self) -> None:
        try:
            if getattr(self, "_read_method", None) is not None:
                return
        except TypeError:  # pydantic bugs with private attr before model_post_init
            pass  # https://github.com/pydantic/pydantic/issues/9098
        tag = "method:"
        fp = str(self.filepath)
        if not fp.startswith(tag):
            self._read_method = self._read
            return
        # Store read method for reuse in subprocesses (where TIMELINES may not be filled)
        # avoid circular import:
        from .data import TIMELINES  # pylint: disable=import-outside-toplevel

        components = urllib.parse.urlparse(fp)
        assert components.netloc == ""
        assert components.params == ""
        assert components.fragment == ""
        # use a specific loader
        inst = TIMELINES[self.timeline]
        kwargs = dict(urllib.parse.parse_qsl(components.query, strict_parsing=True))
        self._read_method = functools.partial(getattr(inst, components.path), **kwargs)

    def read(self) -> tp.Any:
        self._set_read_method()
        return self._read_method()

    @abstractmethod
    def _read(self) -> tp.Any:
        return

    def _missing_duration_or_frequency(self) -> bool:
        return any(not x or pd.isna(x) for x in [self.duration, self.frequency])


class BaseSplittableEvent(BaseDataEvent):
    """
    Base class for dynamic events (audio and video), which can be read in parts.
    We only read the section [offset, offset + duration] of the file.
    Importantly, offset is relative to the event, not the absolute time of the timeline.
    """

    offset: pydantic.NonNegativeFloat = 0.0

    def _split(
        self, timepoints: tp.List[float], min_duration: float | None = None
    ) -> tp.Sequence["BaseSplittableEvent"]:
        """Provided n ordered timepoints to split a the event, returns
        the n + 1 corresponding events corresponding to the sections.
        Timepoints are relative to the event and not the absolute time of the timeline.
        """
        # keep only timepoints that are within the sound duration
        timepoints = [t for t in timepoints if 0 < t < self.duration]
        timepoints = sorted(set(timepoints))
        if min_duration:
            delta_before = np.diff(timepoints, prepend=0)
            delta_after = np.diff(timepoints, append=self.duration)
            timepoints = [
                t
                for t, db, da in zip(timepoints, delta_before, delta_after)
                if db >= min_duration and da >= min_duration
            ]
        timepoints.append(self.duration)

        start = 0.0
        data = dict(self)
        cls = self.__class__
        events = []
        for stop in list(timepoints):
            if start >= stop:
                raise ValueError(
                    f"Timepoints should be strictly increasing (got {start} and {stop})"
                )
            data.update(
                start=self.start + start,
                duration=stop - start,
                offset=self.offset + start,
            )
            events.append(cls(**data))
            start = stop
        return events


class BaseText(Event):
    """
    Base class for text events.
    """

    language: str = ""
    text: str = pydantic.Field("", min_length=1)


class Text(BaseText):
    """Possibly multi-sentence text"""

    language: str = ""
    text: str = pydantic.Field(..., min_length=1)


class Button(Text):
    """"""


class Meg(BaseDataEvent):
    """Brain Meg event"""

    subject: StrCast = ""

    def model_post_init(self, log__: tp.Any) -> None:
        self.subject = self.subject
        if self._missing_duration_or_frequency():
            raw = self.read()
            self.duration = raw.times[-1] - raw.times[0]
            self.frequency = Frequency(raw.info["sfreq"])
            if raw.first_samp > 0 and not self.start:
                start = raw.first_samp / self.frequency
                msg = f"Meg event start for timeline {self.timeline} is 0 while "
                msg += f"raw.first_samp = {raw.first_samp} > 0\n"
                msg += f"(start should have been defined as raw.first_samp / raw.info['sfreq'] = {start})"
                raise ValueError(msg)
        if not self.subject:
            raise ValueError("Missing 'subject' field")
        super().model_post_init(log__)

    def _read(self) -> tp.Any:
        import mne

        with ignore_all():
            return mne.io.read_raw(self.filepath)


class Eeg(Meg):
    """Brain Eeg event"""
