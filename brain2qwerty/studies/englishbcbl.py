# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""EnglishBCBL study (PinetAudio2025): the audio (English) MEG typing dataset.

Identical recording pipeline to the Spanish dataset; only event metadata and
the on-disk layout differ. Vendored from the internal neuralhub definition and
rewired onto the public ``neuralset`` study base via ``spanishbcbl``.
"""

import typing as tp
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .spanishbcbl import Pinet2024Meg, _clean_seq


class PinetAudio2025(Pinet2024Meg):
    """Audio (English) variant of the MEG typing experiment."""

    aliases: tp.ClassVar[tuple[str, ...]] = ("EnglishBCBL",)

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
        "audio_name",
        "sentence_id",
    ]
    PERCEP_CONDITION: tp.ClassVar[str] = "audio"

    @staticmethod
    def _add_language(meta: pd.DataFrame) -> pd.DataFrame:
        meta.loc[meta.type.isin(["Word", "Sentence", "Keystroke"]), "language"] = (
            "english"
        )
        return meta

    @staticmethod
    def _additional_info(struct: dict, trial_id: int) -> dict:
        return {
            "audio_name": struct["audio_name"][trial_id],
            "sentence_id": struct["sentence_id"][trial_id],
            "true_sequence": _clean_seq(struct["sequences"][trial_id]),
        }

    def _get_all_files(self) -> pd.DataFrame:
        # Discover every MEG recording and pair it with its behavioural log.
        # The on-disk layout is DATA/FIF/<subject>/<session>/<file>.fif; subject,
        # session and task are parsed from that path.
        BADS = ["01_12875/250514/01_ses05.fif"]  # known-corrupt recordings to skip

        fif_path = self.path / "DATA" / "FIF"
        fif_filenames = sorted(fif_path.rglob("*.fif"))
        recordings = list()
        for file in fif_filenames:
            if str(file)[len(str(fif_path)) + 1 :] in BADS:
                print(f"Discarded recording files: {str(file)[len(str(fif_path))+1:]}")
                continue
            info = str(file).split("/")
            subject_id, session_dir, file_name = info[-3:]
            if file_name[0] == "S":  # some files are prefixed with the subject letter
                file_name = file_name[1:]
            # task encodes the session, e.g. "...ses05" -> session 5
            task = file_name.split(".")[0].lower()[3:]
            if len(task.split("-")) > 1:  # skip split/partial recordings
                continue
            assert task.startswith(
                "ses"
            ), f"{file_name} Task {task} does not start with ses"
            if "part" in task or "pilot" in task:
                session_no = int(task[3:5] + "01")
            else:
                session_no = int(task[3:])
            sub, part_id = subject_id.split("_")
            subject = "S" + sub
            log = self._get_log_file(self.path, subject, session_no, None)
            recordings.append(
                dict(raw=file, session=session_no, subject=subject, task=task, log=log)
            )
        recordings_df = pd.DataFrame(recordings)
        print(recordings_df[["raw", "log"]].map(lambda x: str(x).split("/")[-1]))
        return recordings_df

    @staticmethod
    @lru_cache
    def _get_log_file(path: Path, sid: str, session_no: int, task: None) -> Path:
        log_path = path / "DATA" / "logs" / sid
        fname = f"**/{sid}-session{session_no}-*.mat"
        log_file = sorted(log_path.rglob(fname))
        if len(log_file) < 1:
            raise FileNotFoundError(
                f"Log file not found for {sid} {session_no}, expected at {log_path / fname}"
            )
        if len(log_file) > 1:
            raise FileExistsError(
                f"Multiple log files for {sid} {session_no}: {log_file}"
            )
        return log_file[0]
