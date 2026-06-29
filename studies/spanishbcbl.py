# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""SpanishBCBL study (Pinet2024Meg / Pinet2024Eeg): the MEG/EEG typing dataset.

Defines the neuralset ``Study`` subclasses that read the raw recordings and their
behavioural logs, align the two, and emit a standardized events frame (Keystroke,
Word, Sentence). Importing the module registers the studies so they resolve by
name (``"Pinet2024Meg"`` / alias ``"SpanishBCBL"``). Built on the public
``neuralset`` study base.
"""

import logging
import typing as tp
from functools import lru_cache
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from dtw import dtw, warp
from mne.io.fiff import Raw  # type: ignore
from scipy.io import loadmat

from neuralset import utils as nutils
from neuralset.events import study

logger = logging.getLogger(__name__)

PUNCTUATION = [
    ",",
    ".",
    "!",
    "?",
    ";",
    ":",
    "-",
    "_",
]


def reassign_trigger_values(
    meg_events_: pd.DataFrame, log_: pd.DataFrame
) -> pd.DataFrame:
    """Recover correct trigger ids for MEG events from the behavioural log.

    The MEG trigger channel is noisy/ambiguous, so we align the MEG event timings
    to the log timings with DTW (matching inter-event intervals), keep only the
    one-to-one matches, and copy the log's trigger values onto those MEG events
    (stored in ``corrected_trigger``; unmatched events stay -1).
    """
    log_["duration"] = np.diff(log_.time.values, prepend=[log_.time.values[0]])
    meg_events_["duration"] = np.diff(
        meg_events_.start.values, prepend=[meg_events_.start.values[0]]
    )

    query = log_["duration"].values
    reference = meg_events_["duration"].values

    alignment = dtw(query, reference, keep_internals=True)
    # alignment.plot(type="threeway")

    wq = warp(alignment, index_reference=False)
    assert len(wq) == len(meg_events_)

    # find which indices in log has one to one mapping with meg events
    unique_idx, count_idx = np.unique(wq, return_counts=True)
    one_to_one_idx = unique_idx[count_idx == 1]
    print(f"{len(one_to_one_idx)/len(unique_idx)} log triggers have one to one mapping")
    assert np.max(one_to_one_idx) < len(log_)

    # find where in meg events has one to one mapping to log indices
    where_one_to_one = np.where(np.isin(wq, one_to_one_idx))[0]
    print(f"{len(where_one_to_one)} meg events have one to one mapping")

    # reassign trigger values
    meg_events_["corrected_trigger"] = -1

    for idx in range(len(where_one_to_one)):
        meg_idx = int(where_one_to_one[idx])
        log_idx = int(one_to_one_idx[idx])

        meg_pair = meg_events_.iloc[meg_idx - 1 : meg_idx + 1]

        meg_events_.loc[meg_pair.index, "corrected_trigger"] = log_.iloc[
            log_idx - 1 : log_idx + 1
        ].trigger.values

    return meg_events_


def _clean_seq(sequence: str) -> str:
    """cleans the sequences taken from either the log files directly
    - removes capital letters, accents, extra spaces, and characters not supported
    - removes functuations
    - fixes typos that are present in the log files based on true sentences

    Args:
        sequence (str): sentence to clean

    Returns:
        str: cleaned sentence
    """
    sequence = sequence.lower()
    sequence = _clean_accents(sequence)
    sequence = " ".join(sequence.split())
    if "\ufeff" in sequence:
        sequence = sequence.replace("\ufeff", "")

    if "jugador" in sequence:
        sequence = sequence.replace("la pelota", "la pelot")

    if "los ninos mayores acaba la escuela" in sequence:
        sequence = "los ninos mayores acaban la escuela"

    for p in PUNCTUATION:
        sequence = sequence.replace(p, "")

    return sequence


# helper functions
def _mat2df(struct: np.ndarray) -> pd.DataFrame:
    """convert matlab structure to pandas dataframe"""
    keys = list(struct.dtype.fields.keys())
    out: tp.List[dict] = list()
    try:
        for t in struct:
            out.append(dict())

            for i, k in enumerate(keys):
                out[-1][k] = t[i]
    except TypeError:  ## only one event in struct for audio version of the task
        out = [{k: struct[k] for k in keys}]
    return pd.DataFrame(out)


def _clean_accents(stim: str) -> str:
    """clean accents from a string
    á, é, í, ó, ú, ü, ñ -> a, e, i, o, u, u, n
    """
    if isinstance(stim, str):
        stim = stim.lower()

        stim = stim.replace("á", "a")
        stim = stim.replace("é", "e")
        stim = stim.replace("í", "i")
        stim = stim.replace("ó", "o")
        stim = stim.replace("ú", "u")
        stim = stim.replace("ü", "u")
        stim = stim.replace("ñ", "n")

    return stim


def _ascii_to_letter(x) -> str:
    out = x if pd.isna(x) else chr(int(x)).lower()
    return out


def _is_left_qwerty(k) -> bool:
    """Is the character on the left side of a QWERTY keyboard?"""
    return isinstance(k, str) and k.lower() in "qwertasdfgzxcv"


def _clean_buttons(char):
    """Uniformize button values."""
    special_chars = {
        "º",
        "»",
        "¼",
        "þ",
        "¡",
        "\x14",
        "ü",
        "¢",
        "ý",
        "¤",
        "¿",
        ".",
        '"',
        "û",
        "\t",
        "[",
        "]",
    }

    MAPPING = {
        " ": "<space>",
        "\xa0": "<space>",
        "\r": "<return>",
        "\x08": "<backspace>",
        "à": "a",
        "ñ": "n",
    }

    if char.isnumeric():
        char = "<number>"
    elif char in special_chars:
        char = "<special>"
    else:
        char = MAPPING.get(char, char)

    return char


class _Pinet2024(study.Study):
    """base class for pinet2024, common functions of different device types"""

    device: tp.ClassVar[str] = "Meg"

    licence: tp.ClassVar[str] = ""
    url: tp.ClassVar[str] = ""
    bibtex: tp.ClassVar[str] = ""
    requirements: tp.ClassVar[tuple[str, ...]] = ("huggingface_hub",)
    description: tp.ClassVar[
        str
    ] = """MEG/EEG recordings collected during a keyboard typing task.
    Experiment description:
    - Experiments are organisised by subject, sessions, and blocks. Each subject has two sessions, each session has three tasks, two blocks of which are typing tasks (block1, block2) and one is tapping task
    - Each trial of each block consists of two phases:
        - perception (is_image==True), where words of each sentence are shown as RSVP
        - production (is_image==False), where participants type the sentence they keep in memory
    - The first two trials of each block are training trials, where participants typed with visual feedback. The rest are test trials (typing with no visual feedback on screen)
    - There are 128 unique test sentences. Train trial sentences do not overlap with test trial sentences.
    - Max sentence length is 8 words.
    Usage:
    - Pinet2024Meg for MEG data, Pinet2024Eeg for EEG data
    - In MEG data, recordings from certain different subject numbers belong to the same person. Refer
      to the data spreadsheet to correct the event dataframe when doing per subject analysis.
    """
    KEPT_COLUMNS: tp.ClassVar[list[str]] = [
        "trial_id",
        "time",
        "duration",
        "pressed",
        "key",
        "trigger",
        "is_percep",
        "is_key",
        "stim",
        "true_sequence",
    ]
    PERCEP_CONDITION: tp.ClassVar[str] = "image"

    # SpanishBCBL recordings (MEG + EEG) live in a single dataset repo on the Hub.
    HF_REPO_ID: tp.ClassVar[str] = "bcbl190626/SpanishBCBL"
    HF_REPO_TYPE: tp.ClassVar[str] = "dataset"

    def _download(self) -> None:
        """Download the SpanishBCBL recordings for this device from the Hugging Face Hub.

        Only this study's modality subtree is fetched into ``self.path`` (``MEG/`` for
        :class:`Pinet2024Meg`, ``EEG/`` for :class:`Pinet2024Eeg`), matching the layout the
        loaders expect (``self.path / "MEG" / "FIF"`` and ``self.path / "EEG" / "EEG"``).
        Downloads resume automatically, so re-running skips files already present.

        The dataset is gated, so you must be authenticated: run ``hf auth login`` once, set
        the ``HF_TOKEN`` environment variable, or pass a token via ``huggingface_hub``.

        Repository: https://huggingface.co/datasets/bcbl190626/SpanishBCBL
        """
        from huggingface_hub import snapshot_download

        # Fetch only this study's modality subtree (plus the dataset README).
        modality = "MEG" if self.device == "Meg" else "EEG"
        allow_patterns = [f"{modality}/*", "README.md"]

        logger.info(
            "Downloading %s (%s) from %s into %s",
            self.HF_REPO_ID,
            modality,
            self.HF_REPO_TYPE,
            self.path,
        )
        snapshot_download(
            repo_id=self.HF_REPO_ID,
            repo_type=self.HF_REPO_TYPE,
            local_dir=self.path,
            allow_patterns=allow_patterns,
        )
        logger.info("Finished downloading %s into %s", self.HF_REPO_ID, self.path)

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        # List all recordings: depends on study structure
        recordings = self._get_all_files()

        for rec in recordings.itertuples():
            yield dict(
                subject=str(rec.subject),
                session=rec.session,
                task=str(rec.task),
            )

    def _get_all_files(self) -> pd.DataFrame:
        """function that gets all recording files,
        implemented in child class"""
        raise NotImplementedError

    @staticmethod
    @lru_cache
    def _get_log_file(path: Path, sid: str, session: int, task: str) -> Path:
        """function that gets a log file given session and task"""
        raise NotImplementedError

    def _add_sentence(self, meta) -> pd.DataFrame:
        sentence_list = []

        word_df = meta[meta.type == "Word"]

        for _, group in word_df.groupby(["trial_id"]):
            # get real sent
            true_sent = group.true_sequence.iloc[0]

            # seq id as unique identifier for true sent
            meta.loc[meta.trial_id == group.trial_id.iloc[0], "sentence"] = true_sent

            for is_percep in group["is_percep"].unique():
                sel_df = group[group["is_percep"] == is_percep]

                if len(sel_df) == 0:
                    print(group)

                sentence = {
                    "type": "Sentence",
                    "start": sel_df.start.iloc[0],
                    "duration": sel_df.start.iloc[-1]
                    + sel_df.duration.iloc[-1]
                    - sel_df.start.iloc[0],
                    "text": true_sent,
                    "trial_id": sel_df.trial_id.iloc[0],
                    "is_percep": is_percep,
                    "dropped_char_per": sel_df.dropped_char_per.iloc[0],
                }

                for col in self.KEPT_COLUMNS:
                    if col not in sentence:
                        sentence[col] = sel_df[col].iloc[0]

                sentence_list.append(sentence)

        sentence_list_df = pd.DataFrame(sentence_list)

        return pd.concat([meta, sentence_list_df], ignore_index=True)

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.Raw:
        """specific load raw"""
        raise NotImplementedError

    def _get_raw_file(self, timeline: dict[str, tp.Any]):
        recs = self._get_all_files()
        sub = recs.subject == timeline["subject"]
        ses = recs.session == timeline["session"]
        tas = recs.task == timeline["task"]
        rec = recs.loc[sub & ses & tas]
        if len(rec) < 1:
            raise RuntimeError(f"Cannot find file for timeline={timeline}")
        elif len(rec) > 1:
            raise RuntimeError(f"More than 1 file found for timeline={timeline}")
        raw_file = rec.iloc[0].raw
        return raw_file

    def _compute_key_event_durations(
        self,
        keys: pd.DataFrame,
        keep_key_releases: bool = False,
        keep_orphan_keys: bool = True,
    ) -> pd.DataFrame:
        """Compute key event duration as the difference between the key press and key release times.

        Parameters
        ----------
        keys:
            DataFrame of key events.
        keep_key_releases:
            If True, keep key release events. The duration of these events will be set to 0.
        keep_orphan_keys:
            If True, keep key events that are not closed. If `keep_key_releases` is True, also keep key
            release events that were not opened.

        Returns
        -------
        DataFrame of updated key events with duration column.
        """
        buffer, rows, durations = dict(), list(), list()

        def update(row, duration):
            rows.append(row)
            durations.append(duration)

        assert "Pressed" in keys.columns, "Pressed is not in columns of keys"
        assert pd.isna(keys.Pressed).sum() == 0, "nan in Pressed columns of keys"
        for row in keys.itertuples():
            if (
                row.Keycode == 13
            ):  # Passthrough for return which doesn't have a key release
                update(row, 0)
            elif row.Pressed == 1:
                if row.Keycode not in buffer:
                    buffer[row.Keycode] = row
                elif keep_orphan_keys:
                    # Remove orphan event and record current event
                    update(buffer.pop(row.Keycode), 0)
                    buffer[row.Keycode] = row
                else:
                    raise ValueError(f"key {row.Keycode} is already in buffer")
            else:
                if row.Keycode in buffer:
                    start_row = buffer.pop(row.Keycode)
                    update(start_row, row.Time - start_row.Time)  # type: ignore
                elif not keep_orphan_keys:
                    continue
                if keep_key_releases:
                    update(row, 0)

        if keep_orphan_keys:
            for key in list(buffer.keys()):
                update(buffer.pop(key), 0)
        if buffer:
            raise ValueError(f"Remaining unclosed events: {buffer.keys()}")

        new_keys = pd.DataFrame(rows)
        new_keys["duration"] = durations

        return new_keys

    @staticmethod
    def _additional_info(struct: dict, trial_id: int) -> dict:
        seq = struct["sequence"][trial_id].strip()
        seq = _clean_seq(seq)
        return {"true_sequence": seq}

    def _read_log(self, log_fname: Path) -> pd.DataFrame:
        """Read log file and format into a pandas DataFrame.

        There are two types of events in the .mat file:
        - "key": a key press ('pressed'=True) or key release ('pressed=False') event. Contains
        information about time and the key that was pressed.
        - "rsvp": an image presentation event (RSVP). Contains information about the time the image was
        presented and the word that was presented.
        """

        # The above code is a Python print statement that is not printing anything. The statement is
        # simply calling the print function without any arguments.
        logger.info("Reading log %s", log_fname)

        mat = loadmat(
            log_fname,
            squeeze_me=True,
            struct_as_record=True,
            chars_as_strings=True,
        )

        # read matlab structure
        struct_ = mat["pr_trials"]
        keys = list(struct_.dtype.fields.keys())
        # struct['sequence'] and struct['sequences] both contain true sentences, but the latter does not contain training trials
        struct = {k: struct_[k].item() for k in keys}

        trial_dicts = list()
        for event_type in ("key", "rsvp"):
            for trial_id, trial in enumerate(struct[event_type]):
                if trial is None:
                    continue
                try:
                    trial = _mat2df(trial)
                except TypeError:
                    continue

                trial["trial_id"] = trial_id
                trial["event_type"] = (
                    self.PERCEP_CONDITION if event_type == "rsvp" else "key"
                )
                for key, val in self._additional_info(struct, trial_id).items():
                    trial[key] = val
                trial_dicts.append(trial)
        df = pd.concat(trial_dicts, ignore_index=True)

        # remove accents from key or rsvp stim
        df.stim = df.stim.apply(_clean_accents)
        df["duration"] = np.nan
        # Force ASCII encoding to remove unsupported characters, e.g. "\ufeff"
        df.stim = df.stim.str.encode("ascii", "ignore").str.decode("ascii")

        # Clean image event info
        idx = df.query(f"event_type=='{self.PERCEP_CONDITION}'").index
        df.loc[idx, "trigger"] = 10
        t_vals = np.asarray(df.loc[idx, "t"].tolist(), dtype=np.float64)
        df.loc[idx, "Time"] = t_vals

        # Clean key event info and compute duration
        if "Keycode" in df.keys():
            keys = df[df.event_type == "key"].copy()
            keys["trigger"] = keys["Keycode"]  # type: ignore
            keys["key"] = keys["Keycode"].apply(_ascii_to_letter)  # type: ignore
            keys = self._compute_key_event_durations(keys)  # type: ignore

            # Add new key events back to image events
            images = df[df.event_type == self.PERCEP_CONDITION]
            df = pd.concat([keys, images], ignore_index=True)  # type: ignore
        else:
            df["key"] = None
            df["Pressed"] = None

        # Compute duration for image events
        df = df.sort_values("Time").reset_index(drop=True)
        df["_duration"] = -df.Time.diff(-1).fillna(0.0)
        image_mask = df.event_type == self.PERCEP_CONDITION
        # pandas>=3 is stricter on subset setitem coercion, so assign aligned float arrays.
        df["duration"] = pd.to_numeric(df.get("duration"), errors="coerce").astype(
            np.float64
        )
        df.loc[image_mask, "duration"] = df.loc[image_mask, "_duration"].to_numpy(
            dtype=np.float64
        )

        # Clean column names
        df["time"] = df["Time"]
        df["pressed"] = df["Pressed"].astype(bool)
        df["is_percep"] = df.event_type == self.PERCEP_CONDITION
        df["is_key"] = df.event_type == "key"
        df = df[self.KEPT_COLUMNS]
        return df

    def _preproc_log(
        self,
        timeline: dict[str, tp.Any],
        match_dropped_chars: bool = True,
    ) -> pd.DataFrame:
        """Format metadata contained into a log file.

        This function reads the log file (.mat) then formats it to extract metadata. Specifically, the
        information about the order of words in trials ('word_id'), and characters in words ('char_id')
        are extracted and aligned between presented and typed words.

        Whitespace characters are matched to the word that follows them, however they are not
        considered when attributing character IDs, i.e. char_id=0 is given to the actual first letter
        of a word.

        Parameters
        ----------
        log_fname :
            Path to the log file (.mat file).
        match_dropped_chars :
            If True, typed characters that were dropped when matching with the groundtruth word
            sequence will be mapped to the word and word_id of the previous character.
        drop_seq_threshold :
            drop entire sequence if the number of dropped letters is above this percentage of the total number of letters in a sequence

        """

        log_fname = self._get_log_file(
            self.path, timeline["subject"], timeline["session"], timeline["task"]
        )
        # Contains both presentation info and actual typed characters
        log = self._read_log(log_fname)

        images = log.query("is_percep")
        # check if multiple stim in each trial: if yes take stim, if not take true_seq as text
        trial = images[images.trial_id == images.trial_id.iloc[0]]
        if len(trial) > 1:
            assert len(trial) == len(
                trial.true_sequence.values[0].split(" ")
            ), f"num of words in seq does not match words, got seq {trial.true_sequence.values[0]}, but stim sequence {trial.stim.values}"

            log.loc[images.index, "text"] = images.stim.str.lower()

            # Add word IDs for image events
            for _, trial in images.groupby("trial_id"):
                log.loc[trial.index, "word_id"] = range(len(trial))  # type: ignore

        elif len(trial) == 1:  # audio version
            log.loc[images.index, "text"] = images.true_sequence.str.lower()
        log["dropped_char_per"] = np.zeros(len(log))

        # Match typed characters with the corresponding presented word
        for _, trial in log.groupby("trial_id"):
            images = trial.query("is_percep")

            # label which word each char belongs to
            if len(images) > 1:
                words = images.text.values
            elif len(images) == 1:  # audio version
                sentence = images.text.values[0]
                words = sentence.split(" ")
                assert len(words) > 1, f"only one word in sentence {sentence}"
                words = np.array([w for w in words if w != ""])

            idx = np.cumsum([c == " " for c in " ".join(words)])

            # query the typed keys
            keys = trial.query("is_key and pressed").query(
                # Ignore backspace and carriage return
                'key not in ("\x08", "\\r")'
            )
            typed = "".join(keys.key.values)

            # cast char into numbers
            presented_unicode = [ord(c) for c in " ".join(words)]
            typed_unicode = [ord(c) for c in typed]
            # match typed vs true char
            i, j = nutils.match_list(presented_unicode, typed_unicode)

            log.loc[keys.index[j], "text"] = words[idx[i]]
            log.loc[keys.index[j], "word_id"] = idx[i]

            dropped_inds = sorted(list(set(range(len(typed))) - set(j)))
            dropped_chars = [typed[i] for i in dropped_inds]
            eps = 1e-8
            # log how many characters have been dropped for this sequence
            log.loc[trial.index, "dropped_char_per"] = len(dropped_chars) / (
                len(typed) + eps
            )

            # Match dropped characters (i.e. typed characters which are not in the true sequence) to
            # the previous word the participant was typing
            if dropped_chars and match_dropped_chars:
                for ind in dropped_inds:
                    if ind == 0:
                        # if the dropped character is the first character of the sequence, we cannot
                        # attribute it to the previous word since the previous event in log is an image event, we
                        # label it as failed
                        log.loc[keys.index[ind], ["text"]] = "FAILED"
                        continue

                    previous_word_and_id = log.loc[
                        keys.index[ind] - 1, ["text", "word_id"]
                    ]
                    log.loc[keys.index[ind], ["text", "word_id"]] = previous_word_and_id

        # Add character IDs
        keys = log.query('is_key and key != " "')
        log["char_id"] = None
        for _, trials in keys.groupby(["trial_id", "word_id"]):
            log.loc[trials.index, "char_id"] = range(len(trials))  # type: ignore

        log["is_left_key"] = False
        keys = log.query("is_key")
        log.loc[keys.index, "is_left_key"] = keys.key.apply(_is_left_qwerty).astype(bool)
        log["trigger"] = log.trigger.astype(float)

        return log

    def _postprocess_meta(self, meta: pd.DataFrame) -> pd.DataFrame:
        """process event df post aligning log file and recording triggers

        - rename columns
        - drop buttons with incomplete information
        - add typed word events

        Args:
            meta (pd.DataFrame): aligned event df

        """
        # Enrich events
        meta["type"] = "Other"
        meta["button"] = meta.key
        meta.loc[meta["is_percep"], "type"] = (
            "Word" if self.PERCEP_CONDITION == "image" else "Sentence"
        )
        meta.loc[meta.is_key & ~meta.key.isna(), "type"] = "Keystroke"
        meta["sequence_id"] = meta.trial_id
        meta["text"] = meta.text.fillna(" ")

        # Figure out button mapping
        buttons = meta[(meta.type == "Keystroke") & (meta.text != " ")].reset_index(
            drop=True
        )
        # empty words (e.g. from <return> characters) are excluded above
        assert not any(buttons.button.isna()), "buttons.button has nan"
        assert all(buttons.button != ""), "buttons.button has empty value"

        buttons.button = buttons.button.apply(_clean_buttons)

        # Filter typed words only
        words = buttons.text

        buttons["word_index"] = np.cumsum(words.shift(1, fill_value=0) != words)
        typed_words = list()
        prev_word = "<start>"
        for (trial_id, word_id), sel in buttons.groupby(["trial_id", "word_id"]):
            if sel.trial_id.nunique() > 1:
                raise ValueError(
                    "selected text contains multiple words from different sequences"
                )
            assert sel.text.nunique() == 1, "selected text contains multiple unique words"
            word = sel.text.values[0]

            if (sel.iloc[0].button == "<space>") & (len(sel) > 1):
                first_button, last_button = sel.iloc[1], sel.iloc[-1]
            else:
                first_button, last_button = sel.iloc[0], sel.iloc[-1]

            start = first_button.start
            duration = last_button.start + last_button.duration - first_button.start
            if word == "":
                word = prev_word
            prev_word = word

            typed_word_dict = {
                "start": start,
                "duration": duration,
                "type": "Word",
                "text": word,
                "trial_id": sel.trial_id.values[0],
                "is_percep": False,
                "word_id": word_id,
                "word_index": sel.word_index.values[0],
                "true_sequence": sel.true_sequence.values[0],
            }

            typed_words.append(typed_word_dict)

        # Concatenate buttons and typed words and images
        events = pd.concat(
            [
                pd.DataFrame(typed_words),
                buttons,
                meta[meta["is_percep"]],
            ],
            ignore_index=True,
        ).reset_index(drop=True)

        # neural set enforce words to have at least 1 char
        bad = (events.type == "Word") & (events.text == "")

        # make sure all events have positive duration
        bad |= events.duration <= 0

        return events.loc[~bad]

    def _align_log_events(
        self, metadata: pd.DataFrame, meg_events: pd.DataFrame, max_n_errors: int = 500
    ) -> pd.DataFrame:
        """Align metadata (from log file) and events (from EEG files).

        Parameters
        ----------
        metadata :
            DataFrame of event metadata obtained from the log file.
        meg_events :
            DataFrame of event metadata obtained from the M/EEG file.
        max_n_errors :
            Raise an error if the number of extra events or missing events reaches this value.

        Returns
        -------
        DataFrame of aligned event metadata.
        """
        # Align metadata and events
        metadata = metadata.query("is_percep or pressed")
        logging.info(
            f"Found {len(metadata)} events in the log file and {len(meg_events)} events in the EEG file."
        )

        i, j = nutils.match_list(meg_events.trigger.values, metadata.trigger.values)
        extra = np.setdiff1d(np.arange(len(meg_events)), i)
        logger.info(f"{len(extra)} extra events")
        missed = np.setdiff1d(np.arange(len(metadata)), j)
        logger.info(f"{len(missed)} missed events")
        # assert (
        # len(extra) < max_n_errors
        # ), f"More than {max_n_errors} extra events."
        assert (
            len(missed) < max_n_errors
        ), f"More than {max_n_errors} missed events ({len(missed)} found)."

        # Check that we identified the word typed at each character
        # assert sum(metadata.char_id == 0) > 250
        mistyped = sum(metadata["is_percep"]) - sum(metadata.char_id == 0)
        # assert mistypped < 80
        logger.info(f"{mistyped} mis-typed words")

        # Merge
        metadata = pd.concat(
            [
                meg_events.iloc[i].reset_index(drop=True),
                metadata.iloc[j].reset_index(drop=True),
            ],
            axis=1,
        )

        metadata = metadata.loc[:, ~metadata.columns.duplicated()]  # type: ignore
        return metadata

    def _preproc_events(self, raw: mne.io.Raw) -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def _add_language(meta: pd.DataFrame) -> pd.DataFrame:
        meta.loc[
            meta.type.isin(["Word", "Sentence", "Keystroke"]),
            "language",
        ] = "spanish"

        return meta

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        # Preproc log file
        log = self._preproc_log(timeline)

        # Preproc mne events — load raw once, reuse for metadata below
        raw = self._load_raw(timeline)
        events = self._preproc_events(raw)

        # Align log and mne events
        meta = self._align_log_events(log, events, max_n_errors=650)
        meta = self._postprocess_meta(meta)

        # add sentence
        meta = self._add_sentence(meta)

        freq = raw.info["sfreq"]
        raw_start = raw.first_samp / freq
        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        # Add MEG event
        raw = dict(
            type=self.device,
            start=raw_start,
            frequency=freq,
            filepath=info,
        )

        df = pd.concat([pd.DataFrame([raw]), meta], ignore_index=True)
        # add language (overridable by subclasses for other languages)
        df = self._add_language(df)

        timeline_id = f"{timeline['subject']}_{timeline['session']}_{timeline['task']}"
        df["sentence_UID"] = df["trial_id"].astype(str) + "_" + timeline_id

        # drop sequence_id column as it is not consistent across timelines and other useless columns
        df = df.drop(
            columns=[
                "sequence_id",
                "word_id",
                "word_index",
                "trigger",
                "time",
                "pressed",
                "key",
                "is_key",
                "stim",
                "char_id",
                "is_left_key",
                "dropped_char_per",
            ]
        )

        df.loc[df.type.isin(["Keystroke", "Word", "Sentence", "Text"]), "modality"] = (
            "typed"
        )

        return df


###Meg
class Pinet2024Meg(_Pinet2024):
    # Public-facing dataset name is "SpanishBCBL"; the class name is kept as
    # Pinet2024Meg because the event-cleaning logic (PrepareSpPinet) keys on
    # subject ids of the form "Pinet2024Meg/S..".
    aliases: tp.ClassVar[tuple[str, ...]] = ("SpanishBCBL",)
    device: tp.ClassVar[str] = "Meg"
    _info: tp.ClassVar[study.StudyInfo | None] = None

    def _get_all_files(self) -> pd.DataFrame:
        """Convenience class to prepare a DataFrame of valid Meg and log filenames for Pinet2023."""
        BADS = [  # Known bad files
            "05_3660/230405/Emptyroom.fif",  # extra file
            "05_3660/230419/block1.fif",  # corrupted file
            "07_10038/230503/testdavid.fif",  # extra file
            "07_10038/230503/testdavid-1.fif",  # extra file
            "18_9228/231116/block1.fif",  # some extra files from sessions
            "20_11966/231122/block2.fif",  # extra file
            "20_11966/231122/block2-1.fif",  # extra file
            # bad files upon visual inspection
            # '03_11123/230313/block1.fif',
            # '03_11123/230313/block1-1.fif',
            # '03_11123/230327/03_11123_block2.fif',
            # '03_11123/230327/03_11123_block2-1.fif',
            # '05_3660/230419/block2.fif',
            # '05_3660/230419/block2-1.fif',
            # '10_3660/231020/block2.fif',
            # '10_3660/231020/block2-1.fif',
            # '10_3660/231023/block1.fif',
            # '10_3660/231023/block1-1.fif',
            # added due to log file missing (robintibor@meta.com)
            "03_11123/230327/03_11123_block1.fif",
            "04_3660/230405/Emptyroom.fif",
            "05_3660/230405/block2.fif",
            "06_10216/230502/block2.fif",
            "06_10216/230502/block3.fif",
            "8_11374/231023/block2_1.fif",
            "8_11374/231023/block2.fif",
            "09_11482/230512/block1.fif",
            # failed session
            "23_9948/240514/block2fail.fif",
            "23_9948/240514/block22.fif",
        ]

        # Find fif files
        fif_path = self.path / "MEG" / "FIF"
        fif_filenames = sorted((fif_path).rglob("*.fif"))
        # n_expected_files = 179
        # assert (
        #     len(fif_filenames) == n_expected_files
        # ), f"Expected {n_expected_files} fif files, got {len(fif_filenames)}"
        recordings = list()
        for file in fif_filenames:
            # skip bad
            if str(file)[len(str(fif_path)) + 1 :] in BADS:
                print(f"Discarded recording files: {str(file)[len(str(fif_path))+1:]}")
                continue

            # get info
            info = str(file).split("/")

            subject_id, session_dir, file_name = info[-3:]

            # handle task
            task = file_name.split(".")[0].lower()
            if (
                "tapping" in task or len(task.split("-")) > 1 or "typing" in task
            ):  # ignore tapping or short second fif file
                continue

            # handle subject
            if len(subject_id.split("_")) > 2:  # handle duplicate dir of subject 2
                continue

            sub, part_id = subject_id.split("_")

            # handle duplicate dir of subject 4 and 14
            if sub == "04" and part_id == "3660":  # wrongly labeled
                continue
            if (
                sub == "14" and part_id == "9876"
            ):  # 14_9875 has the complete files for subject 14
                continue

            subject = "S" + str(int(sub))
            assert isinstance(subject, str)

            # skip S13 left handed
            if subject == "S13":
                continue

            # skip S1 block 2 recording cuz no idea where it's from
            if task == "block2" and subject == "S1":
                continue
            elif task == "block3" and subject == "S1":
                task = "block2"
            elif (
                task == "block2" or task == "block2_1"
            ) and subject == "S21":  # sub21 use block2_2
                continue

            # S14 mispelled block2
            if task == "bolck2":
                task = "block2"

            # handle session, make sure session dir can be converted to int
            assert int(session_dir)
            session = self._retrieve_session(fif_path, subject_id, session_dir)

            # handle subj3 s2 weird naming (03_03_11123_block1.fif)
            if subject == "S3" and session == 2:
                task = task.split("_")[-1]

            # handle subj18 and 20 and 21 for names like block1_1
            if subject == "S18" or subject == "S20" or subject == "S21":
                if len(task.split("_")) > 1:
                    task = task.split("_")[0]

            # Read log
            log = self._get_log_file(self.path, subject, session, task)

            rec = dict(
                raw=file,
                session=session,
                subject=subject,
                task=task,
                log=log,
            )

            recordings.append(rec)

        recordings_df = pd.DataFrame(recordings)
        return recordings_df

    @staticmethod
    @lru_cache
    def _get_log_file(path: Path, sid: str, session: int, task: str) -> Path:
        """Identify log files (.mat) for a specific user ID, session number and task."""

        # Find matching files
        log_path = path / "MEG" / "logs"

        # handle subj1 and 2 reverse labeled log file for session 2 and bad naming
        if sid == "S1" or sid == "S2":
            if "block1" in task.lower():
                task = "-1"
            elif "block2" in task.lower():
                task = "-2"
            elif "block3" in task.lower():
                task = "-2"
            elif task.lower() == "typing_s2":  # bad naming for sub1 session2
                task = "_tapping"
            else:
                task = "_" + task.lower()
                assert task == "_tapping", f"task is not a block nor tapping, got {task}"

            if session == 2:
                if task == "-1":
                    task = "-2"
                elif task == "-2":
                    task = "-1"
        else:  # for other participants the task in file name is in block# format
            task = "_" + task.lower()

        fname = f"**/{sid}-session{session}{task}*.mat"
        out = list(log_path.glob(fname))

        if len(out) < 1:
            raise FileNotFoundError(
                f"Missing subject {sid}, session {session}, task {task}, expected at {log_path}/{fname}."
            )
        elif len(out) > 1:
            raise ValueError(
                f"More than one file found for subject {sid}, session {session}, task {task}:\n{out}"
            )
        else:
            return out[0]

    # helper function
    @staticmethod
    def _retrieve_session(fif_path, subject_id: str, session_dir: str) -> int:
        # get a sorted list of all sessions within a subject dir
        sessions = sorted([child.name for child in (fif_path / subject_id).iterdir()])
        # check if .DS_store is in sessions
        if ".DS" in sessions[0]:
            sessions = sessions[1:]

        session_num = sessions.index(session_dir) + 1

        # handle exception of subject 8 where 2 sessions are stored under two separate subject dir
        if subject_id == "08_11374":
            assert session_dir == "230508"
            session_num = 1
        if subject_id == "8_11374":
            assert session_dir == "231023"
            session_num = 2
        return session_num

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.Raw:
        raw_file = self._get_raw_file(timeline)
        raw = mne.io.read_raw_fif(
            raw_file, preload=False, verbose=False, allow_maxshield=True
        )

        # function to check if meg has sensor info
        def _has_meg_sensor_info(raw):
            for ch in raw.info["chs"]:
                if ch["kind"] == mne.io.constants.FIFF.FIFFV_MEG_CH:
                    if ch["loc"][
                        :3
                    ].any():  # Checks if the first three elements (the 3D location) are not all zeros
                        return True
            return False

        if not _has_meg_sensor_info(raw):
            raise ValueError("meg has no sensor infor")

        return raw

    def _preproc_events(self, raw: Raw) -> pd.DataFrame:
        # get events from raw file
        events_array = mne.find_events(raw, shortest_event=1)

        events = pd.DataFrame()
        events["start"] = ((events_array[:, 0])) / raw.info["sfreq"]
        events["trigger"] = events_array[:, 2]

        return events


###EEG
class Pinet2024Eeg(_Pinet2024):
    device: tp.ClassVar[str] = "Eeg"
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=62,
        num_subjects=20,
        num_events_in_query=2903,
        event_types_in_query={"Eeg", "Keystroke", "Word", "Sentence"},
        data_shape=(61, 995960),
        frequency=1000,
    )

    def _get_all_files(self) -> pd.DataFrame:
        """Convenience class to prepare a DataFrame of valid EEG and log filenames for Pinet2023."""
        BADS = [  # Known bad files
            # file has been renamed, header indicates different name
            "005_DECOMEG_S2_NOID_task1.vhdr",
            # log swap for task2: 'S8_session2_block2_list1.mat'
            "008_DECOMEG_S1_9846_task1.vhdr",
            # log swap for S2: 'S8_session2_block1_list2.mat'
            "008_DECOMEG_S1_9846_task2.vhdr",
            # file has been renamed, 008_DECOMEG_S2bis_9846_task1.vmrk missing
            "008_DECOMEG_S2_9846_task1.vhdr",
            # file has been renamed: 008_DECOMEG_S2bis_9846_task2.vmrk missing
            "008_DECOMEG_S2_9846_task2.vhdr",
            "009_DECOMEG_S1_9949.vhdr",  # log misaligned?
            # something with the log-> the char_id does not work?
            "009_DECOMEG_S1_9949_task1",
            # log file mismatch
            "012_DECOMEG_S1_11481_task1.vhdr",
            # extra file
            "013_DECOMEG_S1_11478_task1.vhdr",
            # log file missing
            "003_DECOMEG_S1_9337_task1.vhdr",
            "003_DECOMEG_S1_9337_task2.vhdr",
            "004_DECOMEG_S2_NOID_task1.vhdr",
            "004_DECOMEG_S2_noid_task2.vhdr",
            # crashed scripts
            "022_DECOMEG_S2_9948_task1.vhdr",
            "022_DECOMEG_S2_9948_task2.vhdr",
        ]

        # Find VHDR files (the main EEG header files)
        fif_path = self.path / "EEG" / "EEG"
        vhdr_filenames = sorted((fif_path).rglob("*_DECOMEG_*.vhdr"))
        # n_expected_files = 88
        # assert (
        #     len(vhdr_filenames) == n_expected_files
        # ), f"Expected {n_expected_files} fif files, got {len(vhdr_filenames)}"
        recordings = list()
        for file in vhdr_filenames:
            # skip bad
            if file.name in BADS:
                print(f"Log-raw.events problem with: {file.name}")
                continue

            # Parse EEG file name
            info = file.name[:-5].split("_")

            # Handle missing task name in a specific file
            if len(info) == 4:
                assert file.name == "009_DECOMEG_S1_9949.vhdr"
                info += ["task1"]
            elif len(info) == 6:
                assert file.name in [
                    "022_DECOMEG_S2_9948_task1_1.vhdr",
                    "022_DECOMEG_S2_9948_task2_2.vhdr",
                ]
                info = info[:5]
                continue

            # skip tapping task
            subject, _, session, _, task = info
            if task == "tapping":
                continue
            assert session.startswith("S"), "Session str must start with S"
            session_num = int(session[1:])
            assert int(subject), "Subject UID must be castable to an int"

            # skip S1: malformed file name
            if subject == "001":
                continue

            # Read log
            try:
                log = self._get_log_file(self.path, subject, session_num, task)
            except (FileNotFoundError, AssertionError):
                print(f"Missing log file for: {file.name}.")
                continue

            rec = dict(
                raw=file,
                session=session_num,
                subject=subject,
                task=task,
                log=log,
            )

            recordings.append(rec)

        recordings_df = pd.DataFrame(recordings)
        return recordings_df

    @staticmethod
    @lru_cache
    def _get_log_file(path, subject, session, task) -> Path:
        """Identify log files (.mat) for a specific user ID, session number and task."""
        # Handle bad task naming
        if task in ("task", "task1"):
            task = "block1"
        elif task == "task2":
            task = "block2"
        else:
            assert task == "tapping"

        # Find matching files
        log_path = path / "EEG" / "logs"
        fname = f"**/S{int(subject)}_session{session}_{task}*.mat"
        out = list(log_path.glob(fname))

        if len(out) < 1:
            raise FileNotFoundError(
                f"Missing subject {subject}, session {session}, task {task}."
            )
        elif len(out) > 1:
            raise ValueError(
                "More than one file found for subject {subject}, session {session}, task {task}:\n{out}"
            )
        else:
            return out[0]

    def _preproc_events(self, raw: Raw) -> pd.DataFrame:
        # 1. Get mne annotations
        annot_list = list()
        for annot in raw.annotations:
            if annot["description"].startswith("Stimulus/"):
                value = int(annot["description"].split("/S")[-1].split()[-1])
                annot_list.append(dict(start=annot["onset"], value=value))
        df = pd.DataFrame(annot_list)

        # 2. Build standard array of mne events
        events_arr = np.zeros((len(df), 3), dtype=int)
        events_arr[:, 0] = raw.info["sfreq"] * df.start
        events_arr[:, 2] = df.value
        # drop the leading 255 warm-up/trial triggers when present
        if (events_arr[0, 2] == 1) and np.array_equal(
            np.diff(events_arr[:255, 2]), np.ones(254)
        ):
            events_arr = events_arr[255:]  # type: ignore
        columns = ["start_sample", "duration_", "trigger"]

        # 3. Convert to dataframe
        events = pd.DataFrame(events_arr, columns=columns)
        events["start"] = events.start_sample.astype(float) / raw.info["sfreq"]
        events["run_id"] = 0
        return events

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.Raw:
        raw_file = self._get_raw_file(timeline)
        raw = mne.io.read_raw_brainvision(raw_file, preload=False)
        montage = mne.channels.make_standard_montage("standard_1005")
        picks = [ch for ch in raw.ch_names if "EOG" not in ch]
        raw = raw.pick(picks)
        raw.set_montage(montage)
        return raw
