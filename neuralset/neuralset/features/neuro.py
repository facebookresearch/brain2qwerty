# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp

import mne
import numpy as np
import pandas as pd
import pydantic
import sklearn.preprocessing
import torch

import neuralset as ns
from neuralset.base import Frequency, TimedArray
from neuralset.infra import MapInfra

from .base import BaseFeature, BaseStatic

logger = logging.getLogger(__name__)
DataframeOrEventsOrSegments = (
    pd.DataFrame | tp.Sequence[ns.events.Event] | tp.Sequence[ns.segments.Segment]
)


def _overlap(
    start1: float,
    duration1: float,
    start2: float,
    duration2: float,
) -> tuple[float, float]:
    """
    Computes the overlap times between two windows
    """
    starts = (start1, start2)
    stops = tuple(s + d for s, d in zip(starts, (duration1, duration2)))
    start = max(starts)
    stop = min(stops)
    return start, max(0, stop - start)


class Meg(BaseFeature):
    """If frequency is set to "native", the frequency used will be the one provided by the Meg event
    filter and resample preprocessing steps can be cached.

    Parameters
    ----------
    baseline :
        If provided as a tuple (start, end), corresponds to the start and end times (in seconds)
        relative to the **beginning of a window** (i.e. NOT relative to the epoch onset as opposed
        to MNE's convention) of the segment to use for baselining.

    Note
    ----
    Produces float32 Tensors
    """

    name: tp.Literal["Meg"] = "Meg"
    event_types: tp.Literal["Meg", "Eeg"] = "Meg"

    frequency: tp.Literal["native"] | float = "native"
    offset: float = 0.0
    baseline: tuple[float, float] | None = None
    pick_types: tuple[str, ...] = pydantic.Field(("meg",), min_length=1)
    sensor_ablation: str | None = None
    apply_proj: bool = False
    filter: tuple[float | None, float | None] | None = None
    apply_hilbert: bool = False
    notch_filter: float | list[float] | None = None
    mne_cpus: int = -1
    infra: MapInfra = MapInfra(
        timeout_min=120,
        gpus_per_node=0,
        cpus_per_task=10,
        version="1",
    )
    scaler: None | tp.Literal["RobustScaler", "StandardScaler"] = None
    clamp: float | None = None

    _channels: tp.Dict[str, int] = {}

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        prev = super()._exclude_from_cls_uid()
        return prev + ["mne_cpus"]

    def _exclude_from_cache_uid(self) -> list[str]:
        prev = super()._exclude_from_cache_uid()
        return prev + ["baseline", "offset", "clamp"]

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        # Update channel mapping to be robust to mne.Raw
        self._channels = {}
        # check baseline
        if self.baseline is not None:
            issue = len(self.baseline) != 2
            issue |= not all(isinstance(b, float) for b in self.baseline)
            issue |= self.baseline[1] <= self.baseline[0]
            if issue:
                msg = f"baseline must be None or 2 floats, got {self.baseline}"
                raise ValueError(msg)

    def prepare(self, obj: DataframeOrEventsOrSegments) -> None:
        """specify how to load and preprocess the event.
        Can be overriden by user.
        """
        from neuralset import helpers

        events: list[ns.events.Meg]
        events = helpers.extract_events(obj, types=self._event_types_helper)  # type: ignore
        # avoid calling super().prepare to avoid loading a cache
        # (through missing preparation) without need:
        self._get_data(events)
        self._prepare_channels(events)
        if events:  # fill missing info manually
            self._missing_default = torch.zeros(len(self._channels))
            freq = self.frequency if self.frequency != "native" else events[0].frequency
            self._effective_frequency = freq

    # NOTE: We use the FIF format to cache MEG as we don't want to discard information such
    # as projectors and channel info which cannot be saved in the faster BrainVision format.
    # However, FIF files take a lot longer to read - if this becomes a bottleneck, we might need to
    # look into using another file format for MEG too.
    @infra.apply(
        item_uid=lambda e: str(e.filepath),
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
        cache_type="MneRawFif",
    )
    def _get_data(self, events: list[ns.events.Meg]) -> tp.Iterator[mne.io.Raw]:
        for event in events:
            raw = event.read()

            raw = raw.pick(self.pick_types, verbose=False)

            if self.notch_filter is not None:
                raw.load_data()
                raw = self._notch_filter(raw, self.notch_filter, self.mne_cpus)

            if self.filter is not None:
                raw.load_data()
                l_freq, h_freq = self.filter
                # Ignore lowpass filter if cutoff is higher than Nyquist frequency
                if h_freq is not None and h_freq > raw.info["sfreq"] / 2:
                    logger.warning(
                        "Lowpass filter cutoff frequency is higher than Nyquist frequency. "
                        "Setting it to None."
                    )
                    h_freq = None

                raw.filter(
                    l_freq,
                    h_freq,
                    picks=("eeg", "emg", "meg"),
                    n_jobs=self.mne_cpus,
                    verbose=False,
                )
            if self.apply_hilbert:
                raw.load_data()
                raw = raw.apply_hilbert(envelope=True)

            freq = event.frequency if self.frequency == "native" else self.frequency
            if freq != event.frequency:
                raw.load_data()
                raw = raw.resample(freq, n_jobs=self.mne_cpus, verbose=False)

            if self.scaler is not None:
                raw.load_data()
                scaler = getattr(sklearn.preprocessing, self.scaler)()
                raw._data = scaler.fit_transform(raw._data.T).T

            if self.apply_proj:
                raw.apply_proj()

            yield raw

    def _get_timed_arrays(
        self, events: list[ns.events.Meg], start: float, duration: float
    ) -> tp.Iterable[TimedArray]:
        for event in events:
            yield self._get_timed_meg(event, start, duration)

    def _get_timed_meg(
        self, event: ns.events.Meg, start: float, duration: float
    ) -> TimedArray:
        start += self.offset

        # Extend window in case of disjoint baseline
        window_start, window_stop = start, start + duration
        if self.baseline is not None:
            if self.baseline[0] >= self.baseline[1]:
                msg = f"unexpected baseline:{self.baseline}"
                raise RuntimeError(msg)
            window_start = min(window_start, start + self.baseline[0])
            window_stop = max(window_stop, start + self.baseline[1])

        # cached_preprocessing
        # (copy to avoid corrupting cache)
        raw = next(self._get_data([event]))
        freq = Frequency(raw.info["sfreq"])

        if not isinstance(raw, mne.io.BaseRaw):  # for typing
            raise TypeError("Output of _get_preprocessed_data should be mne.io.BaseRaw")
        # safeguard for first_samp
        if raw.first_samp and not event.start:
            msg = "event.start should be raw.first_samp / freq for consistency"
            raise RuntimeError(msg)
        overlap_start, overlap_duration = _overlap(
            event.start, event.duration, window_start, window_stop - window_start
        )

        data_start = overlap_start - event.start  # time in the M/EEG referential
        # times in time_as_index are assumed to be relative to first_samp (cf doc)
        # time_as_index is slow, so let's do it manually
        # start_idx, stop_idx = raw.time_as_index([meg_start, meg_start + overlap_duration])
        start_idx = max(0, freq.to_ind(data_start))
        if start_idx == raw.n_times:
            start_idx -= 1
        # apply freq on overlap to keep always the same size, and minimum to 1
        stop_idx = start_idx + max(1, freq.to_ind(overlap_duration))
        try:
            npdata, _ = raw[:, start_idx:stop_idx]
        except ValueError:
            msg = (
                "Failed to read event %r (start=%s duration=%s)\n"
                "(start_idx=%s stop_idx=%s in %s)"
            )
            logger.warning(msg, event, start, duration, start_idx, stop_idx, raw)
            raise
        if npdata.shape[-1] == 0:  # border case
            npdata = np.zeros(npdata.shape[:-1] + (stop_idx - start_idx,))
        tdata = TimedArray(
            frequency=freq,
            duration=overlap_duration,
            start=overlap_start,
            data=np.asarray(npdata).astype(np.float32),
        )

        # Apply baseline to the data
        if self.baseline is not None:
            baseline_duration = self.baseline[1] - self.baseline[0]
            base = tdata.overlap(start + self.baseline[0], baseline_duration)
            if base.data.size:
                tdata.data -= base.data.mean(1, keepdims=True)
        tdata = tdata.overlap(start=start, duration=duration)

        # initialize output
        channel_idx = self._get_channels(raw.ch_names)
        timed_out = TimedArray(frequency=freq, start=start, duration=duration)
        out_shape = (len(self._channels), timed_out.data.shape[-1])
        out = np.zeros(out_shape, dtype=np.float32)
        if tdata.start == start and tdata.duration == duration:
            timed_out = tdata  # bypass copy for efficiency
        else:
            timed_out += tdata
        if self.clamp is not None:
            timed_out.data = np.clip(timed_out.data, a_min=-self.clamp, a_max=self.clamp)
        out[channel_idx, :] = timed_out.data
        timed_out.start -= self.offset
        timed_out.data = out
        return timed_out

    def _update_channels(self, ch_names: list[str]) -> None:
        channels = self._channels  # avoid calling pydantic attr too many times
        for ch in ch_names:
            if ch not in self._channels:
                channels[ch] = len(channels)

    def _prepare_channels(self, events: list[ns.events.Meg]) -> None:
        for raw in self._get_data(events):
            self._update_channels(raw.ch_names)

    def _get_channels(self, ch_names: list[str]) -> list[int]:
        if not self._channels:
            self._update_channels(ch_names)
        try:
            channel_idx = [self._channels[ch] for ch in ch_names]
        except KeyError as e:
            msg = f"Channel {e} not found in the channel mapping, likely because "
            msg += "this dataset contains recordings with different sets of channel "
            msg += "names. Try calling self.prepare on the whole events dataframe."
            raise KeyError(msg) from e
        return channel_idx

    @staticmethod
    def _notch_filter(
        raw: mne.io.Raw, notch_filter: float | list[float], mne_cpus: int
    ) -> mne.io.Raw:
        notch_filter = [notch_filter] if isinstance(notch_filter, float) else notch_filter
        notch_freqs: list[float] = []
        for freq in notch_filter:
            notch_freqs.extend(
                np.arange(freq, min(raw.info["sfreq"] / 2, 301), freq).tolist()  # type: ignore
            )

        if len(notch_freqs) == 0:
            logger.info("Not applying notch filter as no valid frequencies were found.")
        else:
            logger.info("Applying notch filter with notch_freqs=%s", sorted(notch_freqs))
            raw = raw.notch_filter(
                notch_freqs, phase="zero", filter_length="auto", n_jobs=mne_cpus
            )
        return raw


class Eeg(Meg):
    name: tp.Literal["Eeg"] = "Eeg"  # type: ignore
    event_types: tp.Literal["Eeg"] = "Eeg"
    pick_types: tuple[str, ...] = pydantic.Field(("eeg",), min_length=1)


class ChannelPositions(BaseStatic):
    """Channel positions in 2D, extracted from a Raw object's mne.Info.

    Parameters
    ----------
    neuro :
        Feature that defines the preprocessing steps applied to the Raw objects.
        This can either be specified in the config, or built with the `build` method.
    n_spatial_dims :
        Number of spatial dimensions (i.e. coordinates) to extract for each channel. For
        `n_spatial_dims=2`, the 2D projection of the channel positions as obtained through
        `mne.Layout` will be used. For `n_spatial_dims=3`, the 3D positions are extracted from
        `mne.Montage` instead.
    layout_or_montage_name :
        Name of the Layout or Montage to use. See `mne.channels.read_layout()` for a list of valid
        layouts and `mne.channels.get_builtin_montages()` for standard montages. If not provided,
        the function will look for a layout in the `Raw.info` object or for a montage in the `Raw`
        object.
        NOTE: MNE's standard montages are only for EEG systems; MEG montages must be loaded from
              the raw data.
    include_ref_eeg :
        If True, additionally try to extract the position of the anode of bipolar EEG channel (e.g.
        for the channel name "P3-Cz", return position of both "P3" and "Cz"), yielding and output
        of shape (n_channels, n_spatial_dims * 2). If True, `event_types` must be one of
        Eeg.
    normalize :
        If True, min-max normalize channel positions between 0 and 1 across each dimension. If
        False, 2D positions are in arbitrary units given by the mne.Layout projection, while 3D
        positions will be in decimeters (approximately in the range [-1, 1]).
    factor :
        Factor to scale the channel positions by. E.g. set it to 10.0 to get 3D coordinates in
        decimeters, which yields values approximately in the range [-1, 1].
    """

    name: tp.Literal["ChannelPositions"] = "ChannelPositions"
    event_types: tp.Literal["Meg"] = "Meg"
    neuro: tp.Annotated[Meg | Eeg, pydantic.Field(discriminator="name")] | None = None

    n_spatial_dims: tp.Literal[2, 3] = 2
    layout_or_montage_name: str | None = None
    include_ref_eeg: bool = False
    normalize: bool = True
    factor: float = 1.0

    _neuro: Meg | Eeg = pydantic.PrivateAttr()

    # Value to use for channels that are not found in the layout
    INVALID_VALUE: tp.ClassVar[float] = -0.1

    infra: MapInfra = MapInfra()

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if self.neuro is not None:
            if self.include_ref_eeg and self.neuro.name != "Eeg":
                msg = "include_ref_eeg=True is only supported for events_types "
                msg += f"Eeg, got {self.event_types}."
                raise ValueError(msg)
            self._neuro = self.neuro

    def build(self, neuro: Meg | Eeg) -> "ChannelPositions":
        config = self.model_dump()
        config["neuro"] = neuro
        return self.__class__(**config)

    def prepare(self, obj: DataframeOrEventsOrSegments) -> None:
        from neuralset import helpers

        events = helpers.extract_events(obj, types=self._event_types_helper)
        if not hasattr(self, "_neuro"):
            raise ValueError(
                "The neuro feature is not set. Either set it in the config or call build."
            )
        self._neuro.prepare(events)  # Ensure the Raw objects have been precomputed
        super().prepare(events)

    def _get_layout_positions(self, raw: mne.io.Raw) -> dict[str, list[float]]:
        if self.layout_or_montage_name is not None:
            layout = mne.channels.read_layout(self.layout_or_montage_name)
        else:
            try:
                layout = mne.find_layout(raw.info)
            except RuntimeError as err:
                msg = "No valid layout found. Please specify a layout to load with argument "
                msg += "`layout_name` or explicitly set a montage in the study class (e.g. with "
                msg += "`raw.set_montage()`)."
                raise ValueError(msg) from err

        mapping = {name: pos[:2].tolist() for name, pos in zip(layout.names, layout.pos)}
        return mapping

    def _get_montage_positions(self, raw: mne.io.Raw) -> dict[str, list[float]]:
        if self.layout_or_montage_name is not None:
            montage = mne.channels.make_standard_montage(self.layout_or_montage_name)
        else:
            montage = raw.get_montage()
        if montage is None:
            raise RuntimeError(
                "No montage found in the Raw object. Please set a montage in the study class."
            )
        mapping = montage.get_positions()["ch_pos"]
        mapping = {name: pos.tolist() for name, pos in mapping.items()}
        return mapping

    def _get_meg_3d_positions(self, raw: mne.io.Raw) -> dict[str, list[float]]:
        return {ch["ch_name"]: ch["loc"][:3] for ch in raw.info["chs"]}

    def _get_channel_positions_from_raw(self, raw: mne.io.Raw) -> torch.Tensor:
        """Get scaled channel positions for channels in Raw object.

        Returns
        -------
        torch.Tensor :
            Positions for each channel, of shape (n_channels, n_spatial_dims). When including
            reference channel (self.include_ref_eeg is True), output shape is
            (n_channels, n_spatial_dims * 2) where each row contains the coordinates of the cathode
            channel followed by the coordinates of the anode.
        """
        pos_mapping = {}
        if self.n_spatial_dims == 2:
            pos_mapping = self._get_layout_positions(raw)
        elif self.n_spatial_dims == 3:
            if self._neuro.name == "Meg":
                pos_mapping = self._get_meg_3d_positions(raw)
            else:
                pos_mapping = self._get_montage_positions(raw)

        ch_names: list[str] = []
        valid_inds: list[int] = []
        invalid_names: list[str] = []
        ch_index = 0
        for ch_name in raw.ch_names:
            if self.include_ref_eeg:  # Handle bipolar channel names
                names = ch_name.split("-", 1) if "-" in ch_name else [ch_name, None]
            else:
                names = ch_name.split("-")[:1]
            for name in names:
                ch_names.append(name)
                if name in pos_mapping.keys():
                    valid_inds.append(ch_index)
                elif name is not None:
                    invalid_names.append(name)
                ch_index += 1

        if not valid_inds:
            raise ValueError(f"No channel has valid positions: {raw.ch_names}.")

        if len(valid_inds) < 0.1 * ch_index:
            unique_invalid_names = set(invalid_names) - {None}
            msg = f"Fewer than 10% of the channels have valid positions: {unique_invalid_names}."
            logger.warning(msg)

        positions = np.array(
            [
                (
                    pos_mapping[name]
                    if name in pos_mapping
                    else [np.nan] * self.n_spatial_dims
                )
                for name in ch_names
            ]
        )

        if self.normalize:
            ptp = np.nanmax(positions, axis=0, keepdims=True) - np.nanmin(
                positions, axis=0, keepdims=True
            )
            if (ptp == 0.0).any():
                # Can happen if all electrodes are on a same horizontal and/or vertical line
                ptp[ptp == 0.0] = 1.0
            positions = (positions - np.nanmin(positions, axis=0, keepdims=True)) / ptp

        positions *= self.factor  # Scale positions by factor
        positions = np.nan_to_num(positions, nan=self.INVALID_VALUE)

        n_spatial_dims = self.n_spatial_dims
        if self.include_ref_eeg:
            n_spatial_dims *= 2  # type: ignore
            positions = positions.reshape(len(raw.ch_names), n_spatial_dims)

        channel_idx = self._neuro._get_channels(raw.ch_names)
        out = torch.full((len(self._neuro._channels), n_spatial_dims), self.INVALID_VALUE)
        out[channel_idx, :] = torch.from_numpy(positions).float()

        return out

    def _exclude_from_cache_uid(self) -> list[str]:
        ex = super()._exclude_from_cache_uid()
        if not hasattr(self, "_neuro"):
            raise RuntimeError("Should not happen")
        neuro_ex = self._neuro._exclude_from_cache_uid()
        return ex + [f"neuro.{n}" for n in neuro_ex]

    @infra.apply(
        item_uid=lambda e: str(e.filepath),
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
    )
    def _get_data(self, events: list[ns.events.Meg]) -> tp.Iterator[torch.Tensor]:
        if not hasattr(self, "_neuro"):
            raise ValueError(
                "The neuro feature is not set. Either set it in the config or call build."
            )
        for raw in self._neuro._get_data(events):
            yield self._get_channel_positions_from_raw(raw)

    def get_static(self, event: ns.events.Meg) -> torch.Tensor:
        return next(self._get_data([event]))
