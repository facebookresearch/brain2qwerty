# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
import warnings

import numpy as np
import pandas as pd
import pydantic
import torch

import neuralset as ns
from neuralset.base import Frequency as Frequency
from neuralset.base import TimedArray as TimedArray
from neuralset.base import _Module
from neuralset.events import Event, EventTypesHelper
from neuralset.segments import Segment

logger = logging.getLogger(__name__)


class BaseFeature(_Module):
    """Base class for defining features value based on a name.
    The aggregation parameter defines how to merge the values of multiple events.
    """

    event_types: str | tuple[str, ...] = ""
    # eg: event_types: str | tuple[str] = ("Image", "Text")

    aggregation: tp.Literal[
        "single", "sum", "average", "first", "middle", "last", "cat", "stack", "trigger"
    ] = "single"
    # builds feature even when no corresponding event is provided
    allow_missing: bool = False
    frequency: float | tp.Literal["native"] = 0.0
    _effective_frequency: float | None = None
    _CLASSES: tp.ClassVar[dict[str, tp.Type["BaseFeature"]]] = {}
    _event_types_helper: EventTypesHelper

    # internal
    _missing_default: torch.Tensor | None = None

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: tp.Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        # check params
        super().__init_subclass__()
        # add event requirements to the feature requirements
        if not cls._can_be_instanciated():
            return
        model_fields: dict[str, pydantic.FieldInfo] = cls.model_fields  # type: ignore
        event_types: tp.Any = model_fields["event_types"].default  # type:ignore
        name = cls.__name__
        if not event_types:
            msg = f"Default event_types must be specified for {cls.__name__}"
            raise RuntimeError(msg)
        if hasattr(cls, "event_type") or "event_type" in model_fields:
            msg = f"In {name!r}, event_type is deprecated, use event_types instead "
            msg += "as a feature name of tuple of feature names."
            raise RuntimeError(msg)
        # security checks for new _get_data
        legafuncs = [
            "_get_latents",
            "_get_latent",
            "_get_preprocessed_data",
            "_events_to_data",
            "_get_channel_positions",
        ]
        for func in legafuncs:
            if hasattr(cls, func):
                msg = f'In {name!r}, found function {func!r} which should be renamed to "_get_data"'
                raise RuntimeError(msg)
        infrafield = cls.model_fields.get("infra", None)
        if infrafield is not None:
            funcname = infrafield.default._infra_method.method.__name__
            if funcname != "_get_data":
                msg = f'In {name!r}, found infra decorating {funcname!r} it should be "_get_data" by convention'
                raise RuntimeError(msg)
        # security checks for new event_types
        if not isinstance(event_types, str):
            is_tuple = isinstance(event_types, tuple)
            if not (is_tuple and all(isinstance(d, str) for d in event_types)):
                msg = f"In {name!r}, event_types attribute must be a string "
                msg += f"or tuple of string, got {event_types}"
                raise TypeError(msg)
        type_helper = EventTypesHelper(event_types)
        for etype in type_helper.classes:
            cls.requirements = cls.requirements + etype.requirements
        BaseFeature._CLASSES[cls.__name__] = cls
        if "name" not in model_fields or model_fields["name"].default != name:  # type: ignore
            # unfortunately, this field can't be added dynamically so far :(
            # https://github.com/pydantic/pydantic/issues/1937
            indication = f"name: tp.Literal[{name!r}] = {name!r}"
            msg = f"Feature {name} has incorrect/missing name field, add:\n{indication}"
            raise NotImplementedError(msg)

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        self._event_types_helper = EventTypesHelper(self.event_types)
        name = self.__class__.__name__
        if self.frequency != "native" and self.frequency < 0.0:
            msg = f"{name}.frequency is neither 'native' nor >= 0 (got {self.frequency})."
            raise ValueError(msg)
        if not (self.frequency or isinstance(self, BaseStatic)):
            msg = f"{name}.frequency=0 is only allowed for static features (did you mean 'native'?)"
            raise ValueError(msg)

    def _exclude_from_cache_uid(self) -> list[str]:
        # feature convention from inheriting cache uid exclusion list
        return ["aggregation", "allow_missing"]

    def prepare(
        self, obj: pd.DataFrame | tp.Sequence[Event] | tp.Sequence[Segment]
    ) -> None:
        """Run _get_data on all events to cache results,
        then call the feature on a single event to populate the shape.

        Parameter
        ---------
        obj: DataFrame or list of events/segments
            The structure containing all the events, be it as a dataframe or list of events or list
            of segments. If you are calling prepare on several objects, then consider avoiding
            DataFrame as this will require more computation.
        """
        from neuralset import helpers

        events = helpers.extract_events(obj, types=self._event_types_helper)
        if self.frequency == "native" and events and hasattr(events[0], "frequency"):
            freqs = set(e.frequency for e in events)  # type: ignore
            cls = self.__class__.__name__
            if len(freqs) > 1:
                msg = f"frequency='native' in {cls} with several different frequencies: {freqs}"
                msg += "\n(all data will not be processing at the same frequency, "
                msg += "should you set the feature frequency?"
                logger.warning(msg)
            elif len(freqs) == 1:
                cls = self.__class__.__name__
                freq = list(freqs)[0]
                msg = f"Processing to native frequency in {cls}.prepare: {freq}Hz"
                logger.info(msg)
        self._get_data(events)
        if events:  # run feature on 1 event to populate shape
            self(
                events[0],
                start=events[0].start,
                duration=0.001,
                trigger=events[0].to_dict(),
            )

    def _get_data(self, events: list[Event]) -> tp.Iterable[tp.Any]:
        """Put heavy computation steps here, and cache the result using exca.MapInfra"""
        for _ in events:
            yield None

    def _get_timed_arrays(
        self, events: list[Event], start: float, duration: float
    ) -> tp.Iterable[TimedArray]:
        raise NotImplementedError

    def __call__(
        self,
        events: tp.Any,  # too complex: pd.DataFrame | list | dict | ns.events.Event,
        start: float,
        duration: float,
        trigger: float | dict[str, tp.Any] | None = None,
    ) -> torch.Tensor:
        """events: the single event (dict | ns.events.Event) or the series
        of events (list of Events | pd.DataFrame) describing the events, each
        containing start and duration.
        start: the start of the segment in the same timeline as the event.
        duration: the duration of the segment.
        """
        _input_events = events

        from neuralset import helpers

        # Check argument
        assert duration >= 0.0, f"{duration} must be >= 0."
        event_types = self._event_types_helper.classes
        name = self.__class__.__name__
        if self.aggregation == "trigger":
            type_ = trigger.get("type", None) if isinstance(trigger, dict) else trigger
            t: tp.Any = trigger
            if type_ in Event._CLASSES:  # convert to event if possible
                t = Event.from_dict(trigger)
            if not isinstance(t, event_types):  # clear error message
                aggregation = self.aggregation
                msg = f"Feature {name} has {aggregation=} but trigger is {t!r} (not {event_types})"
                raise ValueError(msg)
            events = [t]
        events = helpers.extract_events(events, types=self._event_types_helper)
        # create an empty event if nothing is available
        if not events and self.allow_missing and self._missing_default is not None:
            if self._effective_frequency is None:
                msg = f"_missing_default was set for {name} but _effective_frequency is missing"
                raise RuntimeError(msg)
            default = self._missing_default
            freq = Frequency(self._effective_frequency)
            if freq:
                n_times = max(1, freq.to_ind(duration))
                reps = [1 for _ in range(default.ndim)] + [n_times]
                default = default.unsqueeze(-1).repeat(reps)
            return default

        if not events:
            found_types = {type(e) for e in _input_events}
            msg = f"No {event_types} found in segment for feature {name} "
            msg += f"(types found: {found_types} in {_input_events}) "
            if not self.allow_missing:
                msg += f"(filter invalid segments or set allow_missing=True to {name})"
            else:
                msg += "and feature shape not populated "
                msg += '(you may need to call "prepare" on the feature).'
            raise ValueError(msg)

        # Extract value for each relevant event
        if self.aggregation in ("first", "trigger", "single"):
            if self.aggregation == "single" and len(events) > 1:
                msg = f"Found {len(events)} events in the segment but expected only one "
                msg += f"since {name}.aggregation='single'."
                msg += "Update it to sum/average/first/trigger/... ?\n"
                msg += f"{events=}"
                raise ValueError(msg)
            events = events[:1]
        elif self.aggregation == "last":
            events = events[-1:]
        elif self.aggregation == "middle":
            events = [events[len(events) // 2]]
        tarrays = list(
            self._get_timed_arrays(events=events, start=start, duration=duration)
        )
        if self._effective_frequency is None:
            if self.frequency == "native":
                self._effective_frequency = tarrays[0].frequency
            else:
                self._effective_frequency = self.frequency
        # aggregate arrays
        time_info: dict[str, tp.Any] = {
            "start": start,
            "frequency": self._effective_frequency,
            "duration": duration,
        }
        aggreg = "sum"
        if self.aggregation == "average" and len(tarrays) > 1:
            aggreg = self.aggregation
        if self.aggregation not in ("cat", "stack"):
            out = TimedArray(aggregation=aggreg, **time_info)
            for ta in tarrays:
                out += ta
        else:
            arrays = []
            for ta in tarrays:
                out = TimedArray(**time_info)
                out += ta
                arrays.append(out.data)
            func = np.concatenate if self.aggregation == "cat" else np.stack
            data = func(arrays, axis=0)
            out = TimedArray(data=data, **time_info)
        tensor = torch.from_numpy(out.data)
        if not tensor.ndim:
            tensor = tensor.unsqueeze(0)
        # record shape and return
        if self._missing_default is None:
            # last dimension is time if frequency is not 0
            shape = tuple(tensor.shape[: -1 if self.frequency else None])
            self._missing_default = torch.zeros(*shape, dtype=tensor.dtype)
        return tensor

    def _events_from_dataframe(self, events: pd.DataFrame) -> list[tp.Any]:
        # we're loosing type here :(
        from neuralset import helpers  # avoid circular imports

        warnings.warn(
            "_events_from_dataframe is deprecated, use ns.helpers.extract_events instead",
            DeprecationWarning,
        )
        events_ = helpers.extract_events(events, types=self._event_types_helper)
        return events_


class BaseStatic(BaseFeature):
    frequency: float = 0.0

    def get_static(self, event: ns.events.Event) -> torch.Tensor:
        """retrieve the static embedding"""
        raise NotImplementedError

    def _get_timed_arrays(
        self, events: list[Event], start: float, duration: float
    ) -> tp.Iterable[TimedArray]:
        for event in events:
            embedding = self.get_static(event)
            ta = TimedArray(
                frequency=0,
                duration=event.duration,
                start=event.start,
                data=embedding.numpy(),
            )
            yield ta


class LabelEncoder(BaseStatic):
    """Encode a given field from an event, e.g. to be used as a label.

    Parameters
    ----------
    event_types :
        Type of event to apply this feature to.
    event_field :
        Field to encode from the event.
    return_one_hot :
        If True, return one-hot representation of the index. Otherwise, return an int in
        [0, n_unique_values - 1].
    predefined_mapping : Optional dict
        If provided, use this mapping from label to index instead of computing it from data.
    """

    name: tp.Literal["LabelEncoder"] = "LabelEncoder"
    event_types: str | tuple[str, ...] = "Event"
    event_field: str
    return_one_hot: bool = False
    predefined_mapping: dict[str, int] | None = None

    _label_to_ind: dict[str, int] = {}
    _n_classes: int = 0

    def _extract_event_field(self, event: ns.events.Event) -> str:
        """Get the event field value from the event."""
        if hasattr(event, self.event_field):
            return getattr(event, self.event_field)
        else:
            return event.extra[self.event_field]

    def prepare(
        self, obj: pd.DataFrame | tp.Sequence[Event] | tp.Sequence[Segment]
    ) -> None:
        from neuralset import helpers

        events = helpers.extract_events(obj, types=self._event_types_helper)
        field = self.event_field
        if not all(hasattr(e, field) or field in e.extra for e in events):
            msg = f"Field {field} not found in events for {self.__class__.__name__}"
            raise TypeError(msg)

        labels = set(self._extract_event_field(e) for e in events)
        if len(labels) < 2:
            logger.warning(
                f"LabelEncoder has only found one label: {labels}. "
                "This was probably not intended."
            )

        if self.predefined_mapping:
            assert all(
                label in self.predefined_mapping for label in labels
            ), "Some labels in the data are missing from the predefined_mapping."
            self._label_to_ind = self.predefined_mapping
        else:
            self._label_to_ind = {label: i for i, label in enumerate(sorted(labels))}

        self._n_classes = len(set(self._label_to_ind.values()))
        expected_indices = set(range(self._n_classes))
        actual_indices = set(self._label_to_ind.values())
        if expected_indices != actual_indices:
            logger.warning(
                f"Label indices are not contiguous. Expected indices: {expected_indices}, "
                f"but got: {actual_indices}. "
                "This may cause issues with one-hot encoding or class-based operations."
            )

        if events:
            self(events[0], events[0].start, duration=0.001, trigger=events[0].to_dict())

    def get_static(self, event: ns.events.Event) -> torch.Tensor:
        if not self._label_to_ind:
            msg = "Must call label_encoder.prepare(events) before using the feature."
            raise ValueError(msg)
        inds = [self._label_to_ind[self._extract_event_field(event)]]
        label = torch.tensor(inds, dtype=torch.long)
        if self.return_one_hot:
            label = torch.nn.functional.one_hot(label, num_classes=self._n_classes)
        return label
