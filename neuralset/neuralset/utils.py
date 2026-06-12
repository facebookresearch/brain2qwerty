# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import hashlib
import os
import re
import typing as tp
import warnings
from pathlib import Path

import numpy as np


def match_list(A, B, on_replace="delete"):
    """Match two lists of different sizes and return corresponding indice
    Parameters
    ----------
    A: list | array, shape (n,)
        The values of the first list
    B: list | array: shape (m, )
        The values of the second list
    Returns
    -------
    A_idx : array
        The indices of the A list that match those of the B
    B_idx : array
        The indices of the B list that match those of the A
    """
    from Levenshtein import editops  # type: ignore

    if not isinstance(A, str):
        unique = np.unique(np.r_[A, B])
        label_encoder = dict((k, v) for v, k in enumerate(unique))

        def int_to_unicode(array: np.ndarray) -> str:
            return "".join([str(chr(label_encoder[ii])) for ii in array])

        A = int_to_unicode(A)
        B = int_to_unicode(B)

    changes = editops(A, B)
    B_sel = np.arange(len(B)).astype(float)
    A_sel = np.arange(len(A)).astype(float)
    for type_, val_a, val_b in changes:
        if type_ == "insert":
            B_sel[val_b] = np.nan
        elif type_ == "delete":
            A_sel[val_a] = np.nan
        elif on_replace == "delete":
            # print('delete replace')
            A_sel[val_a] = np.nan
            B_sel[val_b] = np.nan
        elif on_replace == "keep":
            # print('keep replace')
            pass
        else:
            raise NotImplementedError
    B_sel = B_sel[np.where(~np.isnan(B_sel))]
    A_sel = A_sel[np.where(~np.isnan(A_sel))]
    assert len(B_sel) == len(A_sel)
    return A_sel.astype(int), B_sel.astype(int)


ISSUED_WARNINGS = set()


def warn_once(message: str) -> None:
    if message not in ISSUED_WARNINGS:
        warnings.warn(message)
        ISSUED_WARNINGS.add(message)


def compress_string(file_) -> str:
    def hash_(s: str) -> str:
        return hashlib.sha256(s.encode()).hexdigest()[:10]

    # if file is a path, hash the parent folder in case
    # several files have the same name in different folders
    file_ = str(file_)
    fname = Path(file_).name

    pattern = r"[^a-zA-Z0-9.\-_]"
    valid = re.sub(pattern, "", fname)

    if len(fname) > 70:
        valid = "_".join([valid[:20], hash_(fname), valid[-20:]])

    folder = str(Path(file_).parent)
    if folder != "." or valid != fname:
        valid = f"{hash_(file_)}_{valid}"

    return valid


# Define a dummy context manager to suppress output
@contextlib.contextmanager
def ignore_all() -> tp.Iterator[None]:
    with open(os.devnull, "w", encoding="utf8") as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield
