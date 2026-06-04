# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
import dataclasses
import logging
import typing as tp
import warnings

import numpy as np
import pandas as pd
import tqdm

from .events import Event
from .utils import warn_once

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Segment:
    """What gets out of a `ns.segments.list_segments(events, duration=1.)`.
    Fields:
    - start: float (start time of the time segment)
    - duration: float (duration of the time segment)
    - ns_events: list of ns.events.Event instances overlapping with this segment

    Additionally there is a lazily computed and "events" property
    as a dataframe of events occuring during this segment.
    """

    # adding a dict interface to this class confuses pytorch-lightning
    # so better avoid it and keep to a standard dataclass

    start: float
    duration: float
    _index: np.ndarray  # indices of the events in the original dataframe
    ns_events: tp.List[Event] = dataclasses.field(default_factory=list)
    _trigger: float | tp.Dict[str, tp.Any] | None = None  # handle differently?

    @property
    def events(self) -> pd.DataFrame:
        """events occuring whithin the segment, as a dataframe"""
        if not self.ns_events:
            raise RuntimeError(f"ns_events was not populated in {self}")
        if len(self.ns_events) != len(self._index):
            msg = f"Cannot recreate events dataframe as some rows were not actual Event\n(on segment={self})"
            raise RuntimeError(msg)
        return pd.DataFrame(index=self._index, data=[e.to_dict() for e in self.ns_events])

    @property
    def event_list(self) -> list[Event]:
        raise RuntimeError(
            "segment.event_list is deprecated in favor of segment.ns_events"
        )

    @property
    def stop(self) -> float:
        return self.start + self.duration

    def _to_feature(self) -> dict[str, tp.Any]:
        """Convenience function for extracting segment information to feature call"""
        return {
            "start": self.start,
            "duration": self.duration,
            "events": self.ns_events,
            "trigger": self._trigger,
        }


def _validate_event(event: pd.Series) -> dict[str, tp.Any]:
    """Validate event, i.e. check fields and values are as expected,
    and update it accordingly.

    This is done by instantiating an event object of the corresponding
    type, which carries out the validation, and then updating the input
    with the applied changes (if any).
    """
    # Check types are valid
    event_type = event["type"]
    lower = {x.lower() for x in Event._CLASSES}
    if event_type in Event._CLASSES:
        event_class = Event._CLASSES[event_type]
        event_obj = event_class.from_dict(event).to_dict()

        # Add back fields that were ignored by the Event class
        # segment.update(asdict(event_obj))
        # Very slow, use dict updating instead
        event_dict = {**event, **event_obj}
    elif event_type in lower:
        raise ValueError(f"Legacy uncapitalized event {event}")
    else:
        warn_once(
            f'Unexpected type "{event["type"]}". Support for new event '
            "types can be added by creating new `Event` classes in "
            "`neuralset.events`."
        )
        event_dict = {**event}

    return event_dict


def validate_events(events: pd.DataFrame) -> pd.DataFrame:
    """Validate the DataFrame of events (not inplace).

    Returns
    -------
    pd.DataFrame
        DataFrame in which each row has been validated and updated.
    """
    if events.empty:
        return events.copy()
    msg = 'events DataFrame must have a "type" column with strings'
    if "type" not in events.keys():
        raise ValueError(msg)
    types = events["type"].unique()
    if not all(isinstance(typ, str) for typ in types):
        raise ValueError(msg)
    # event-level validation
    df = pd.DataFrame(
        events.apply(_validate_event, axis=1).tolist(),
        index=events.index,
    )
    # check for null duration
    null = df.loc[df.duration <= 0, :]
    if not null.empty:
        types = null["type"].unique()
        msg = f"Found {len(null)} event(s) with null duration (types: {types})"
        warnings.warn(msg)
    # sort
    dfs = []
    for _, sub in df.groupby(by="timeline", sort=False):
        dfs.append(
            sub.sort_values(
                by=["start", "duration"], ascending=[True, False], ignore_index=True
            )
        )
    important = ["type", "start", "duration", "timeline"]
    df = pd.concat(dfs, ignore_index=True)
    # reorder columns
    columns = important + [c for c in df.columns if c not in important]
    df = df.loc[:, columns]
    # add dynamic field
    df = df.assign(stop=lambda x: x.start + x.duration)
    return df


def _prepare_strided_windows(
    start: float,
    stop: float,
    stride: float,
    duration: float,
    drop_incomplete: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare strided windows.

    Parameters
    ----------
    drop_incomplete:
        If True, drop windows that are not fully contained within (start, stop). This is the
        behaviour of `mne.events_from_annotations` with `chunk_duration` other than None.
    """
    eps = 1e-8
    if drop_incomplete:
        stop -= duration
    starts = np.arange(start, stop + eps, stride)
    durations = np.full_like(starts, fill_value=duration)
    return starts, durations


def iter_segments(
    events: pd.DataFrame,
    idx: int | pd.Series | None = None,
    *,
    start: float = 0.0,
    duration: float | None = None,
    stride: float | None = None,
    stride_drop_incomplete: bool = True,
) -> tp.Iterator[Segment]:
    """
    Make an iterator of segments based on specific events (`idx`), a `stride`, or both.

    See `ns.segments.list_segments` for a description of parameters.
    """
    df = events
    starts: tp.Any  # to avoid messy type checks with pandas
    durations: tp.Any
    if not hasattr(df, "stop"):  # should be populated
        raise ValueError("Run ns.segments.validate_data on dataframe first")
    if not isinstance(start, (int, float)):
        raise TypeError("start must be int/float")

    creators = SegmentCreator.from_obj(events)
    # Regular division of timeline / event
    start = float(start)
    if idx is None and stride is None:
        # make a big enough stride such that it only creates 1 segment per timeline
        stride = 2 * (1 + abs(start) + max(c.stops.max() for c in creators.values()))
    if stride is not None:
        if not isinstance(stride, (int, float)):
            raise RuntimeError(
                f"stride can only be None or int/float, got {type(stride)}"
            )
        if not isinstance(duration, (int, float)):
            raise RuntimeError(
                f"duration must be int/float for strided windows, got {duration}"
            )
    if idx is None:
        if stride is None or duration is None:
            raise ValueError("Either stride or idx must be provided")
        stride = float(stride)
        duration = float(duration)
        for creator in creators.values():
            starts, durations = _prepare_strided_windows(
                creator.starts.min() + start,
                creator.stops.max() + start,
                stride,
                duration,
                drop_incomplete=stride_drop_incomplete,
            )
            for start_, duration_ in zip(starts, durations):
                seg = creator.select(start=start_, duration=duration_)
                seg._trigger = start_
                yield seg
        return

    # from now on, idx is not None
    # ensure index is a pd.Series of integers
    if isinstance(idx, int):
        idx = df.index == idx  # type: ignore
    if not np.any(idx):
        avail = pd.unique(df["type"])
        raise ValueError(
            "Empty trigger events provided to list_segments (first argument)\n"
            f"Available events.type: {avail} (did you forget capitalizing the event name?)"
        )
    # convert value-based index to boolean-based index
    # caution: "idx.dtype is bool" doesnt work anymore when reloaded
    # from parquet, which gets a weird type-like object as type
    if "bool" in str(idx.dtype).lower():  # type: ignore
        idx = df.loc[idx].index  # type: ignore
    # check index
    df.loc[idx]  # pylint: disable=pointless-statement

    triggers: tp.Generator | list | np.ndarray
    groups = tqdm.tqdm(
        df.groupby("timeline", sort=False), desc="Creating segments", mininterval=4
    )

    for tl_name, tl in groups:
        if not isinstance(tl_name, str):
            raise TypeError(f"timeline should be a string, got {tl_name!r}")
        # If we select the batch based on existing events
        if idx is not None:
            j = tl.index.isin(idx)
            if not any(j):
                # Can happen if timeline does not contain any event of interest
                warn_once(f"No valid events found for timeline {tl_name}.")
                continue

            if stride is None:
                starts = tl.loc[j].start + start
                # cant pickle named tuples, and iterrows is too slow:
                triggers = (r._asdict() for r in tl.loc[j].itertuples())  # type: ignore

                # If duration is not specified, use the duration of the selected events
                if duration is None:
                    durations = tl.loc[j].duration
                else:
                    durations = np.ones_like(starts) * duration

            else:  # Extract sliding windows within each event
                starts, durations, triggers = [], [], []
                for row in tl.loc[j].itertuples():
                    if not isinstance(duration, (int, float)):
                        msg = f"Unsupported type for one of duration {duration}"
                        raise TypeError(msg)
                    _starts, _durations = _prepare_strided_windows(
                        row.start + start,  # type: ignore
                        row.stop + start,  # type: ignore
                        stride,
                        duration,
                        drop_incomplete=stride_drop_incomplete,
                    )
                    starts.append(_starts)
                    durations.append(_durations)
                    triggers.extend([row._asdict()] * len(_starts))  # type: ignore
                starts = np.concatenate(starts)
                durations = np.concatenate(durations)
        # add triggers and events
        creator = creators[tl_name]
        for start_, duration_, trigger_ in zip(starts, durations, triggers):
            seg = creator.select(start=start_, duration=duration_)
            seg._trigger = trigger_
            yield seg


def list_segments(  # pylint: disable=unused-argument
    events: pd.DataFrame,
    idx: pd.Series | None = None,
    *,
    start: float = 0.0,
    duration: float | None = None,
    stride: float | None = None,
    stride_drop_incomplete: bool = True,
) -> list[Segment]:
    """
    Make a list of segments:
    - based on specific events (a single segment is extracted by event):
        ns.segments.list_segments(df, idx=df.type == "Image")
    - based on sliding windows (entire timeline will be subdivided into potentially
        overlapping segments):
        ns.segments.list_segments(df, stride=1.5, duration=3.)
    - or based on both a list of segments and sliding windows (each event will be subdivided
        into potentially overlapping segments; a window must be fully overlapping with the event
        to be valid):
        df.ns.list_segments(df, idx=df.type == "Image", stride=1.5, duration=3.)

    Parameters
    ----------
    idx: pd.Series
        If provided, list of events to use for defining the segments.
    start: float
        Start time (in seconds) of the segment, with respect to the reference event (or stride).
        E.g. use -1.0 if you want the segment to start 1s before the event.
    duration: optional float
        Duration (in seconds) of the segment (defaults to event duration if only using `idx` to
        extract segments based on specific events).
    stride: optional float
        Stride (in seconds) to use to define sliding window segments.
    stride_drop_incomplete: optional bool
        If True and stride is not None, drop segments that are not fully contained within the
        (start, stop) block.
    """
    return list(iter_segments(**locals()))


def find_incomplete_segments(
    segments: tp.Sequence[Segment], event_types: tp.Sequence[tp.Type[Event]]
) -> tp.List[int]:
    """
    Return the indices of segments that do not contain one of the specified event types.
    """
    all_invalid_indices = set()
    for event_type in event_types:
        invalid_indices = set()
        subclasses = [
            name for name, cls in Event._CLASSES.items() if issubclass(cls, event_type)
        ]
        for i, segment in enumerate(segments):
            if not any(e.type in subclasses for e in segment.ns_events):
                invalid_indices.add(i)
        if invalid_indices:
            msg = f"{len(invalid_indices)} segments out of {len(segments)} did not contain valid events for event type {event_type}"
            logger.warning(msg)
        all_invalid_indices.update(invalid_indices)
    return sorted(list(all_invalid_indices))


class SegmentCreator:
    """Given all events in a single timeline, this class extracts segments quickly.
    extract a segment.
    """

    # see profiling for optimization rationale

    def __init__(self, events: list[Event]) -> None:
        timelines = {e.timeline for e in events}
        if len(timelines) > 1:
            name = self.__class__.__name__
            msg = f"Cannot create {name} on several timelines, got {timelines}"
            raise ValueError(msg)
        self.events = np.array(events)
        self.starts = np.array([e.start for e in events])
        self.indices = np.array([e._index for e in events])
        self.stops = np.array([e.duration for e in events]) + self.starts

    @classmethod
    def from_obj(cls, obj: tp.Any) -> dict[str, "SegmentCreator"]:
        """Builds a dictionary with timelines as keys and corresponding
        SegmentCreator as values"""
        from neuralset import helpers

        timeline_events: dict[str, list[Event]] = collections.defaultdict(list)
        for e in helpers.extract_events(obj):
            timeline_events[e.timeline].append(e)
        timelines = list(timeline_events)
        if isinstance(obj, pd.DataFrame):
            # make sure we get all timelines event for empty events
            timelines = list(obj.timeline.unique())
        return {tl: cls(timeline_events[tl]) for tl in timelines}

    def select(self, start: float, duration: float) -> Segment:
        """Create a segment populated with the corresponding events

        Parameters
        ----------
        start: float
            start time of the segment
        duration: float
            duration of the segment
        """
        # strict
        start = float(start)  # avoid np.float (longer prints)
        duration = float(duration)
        select = self.starts < start + duration
        select &= self.stops > start
        events = list(self.events[select])
        index = self.indices[select]
        return Segment(ns_events=events, start=start, duration=duration, _index=index)
