# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import typing as tp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import neuralset as ns
from neuralset.extractors import BaseExtractor
from neuralset.extractors.base import BaseStatic
from neuralset.extractors.neuro import ChannelPositions as _ChannelPositions

# Character vocabulary for the CTC head: a..z plus space ("&"); the blank
# symbol ("-") is class 0.
key_to_int = {
    "a": 1,
    "b": 2,
    "c": 3,
    "d": 4,
    "e": 5,
    "f": 6,
    "g": 7,
    "h": 8,
    "i": 9,
    "j": 10,
    "k": 11,
    "l": 12,
    "m": 13,
    "n": 14,
    "o": 15,
    "p": 16,
    "q": 17,
    "r": 18,
    "s": 19,
    "t": 20,
    "u": 21,
    "v": 22,
    "w": 23,
    "x": 24,
    "y": 25,
    "z": 26,
    "&": 27,
}
letters_withblank = ["-"] + list(key_to_int.keys())

# RoBERTa-large scores the semantic error rate; override with a local cache path.
ROBERTA_PATH = os.environ.get("BRAIN2QWERTY_ROBERTA", "roberta-large")


def build_mlp(
    input_dim: int, output_dim: int, num_layers: int = 1, hidden_dim: int | None = None
) -> nn.Sequential:
    """Stack of ``num_layers`` (Linear, LayerNorm, GELU) blocks."""
    mid = hidden_dim or output_dim
    layers: list[nn.Module] = []
    in_d = input_dim
    for i in range(num_layers):
        out_d = output_dim if i == num_layers - 1 else mid
        layers.extend([nn.Linear(in_d, out_d), nn.LayerNorm(out_d), nn.GELU()])
        in_d = out_d
    return nn.Sequential(*layers)


def compute_output_lens(
    network: torch.nn.Module, neuro_sizes: torch.Tensor
) -> torch.Tensor:
    """Map input MEG lengths to encoder output lengths (post temporal downsampling)."""
    if hasattr(network, "compute_output_lens"):
        return network.compute_output_lens(neuro_sizes)
    conv = network.temporal_downsampling.agg
    return (neuro_sizes - conv.kernel_size[0]) // conv.stride[0] + 1


def apply_jitter(
    data: torch.Tensor, seg: ns.segments.Segment, feat: BaseExtractor
) -> torch.Tensor:
    """Drop a random prefix (up to the pre-trigger window) to jitter sentence onset."""
    seg_start = seg.trigger.start - seg.start
    jitter_amount = np.random.uniform(0, seg_start * feat.frequency)
    return data[:, int(jitter_amount) :]


# --- Hard DTW for word-level contrastive matching --------------------------
@torch.no_grad()
def hard_dtw_path(cost: torch.Tensor) -> list[tuple[int, int]]:
    """Standard DTW with backtracking; returns a monotonic alignment path."""
    N, M = cost.shape
    D = cost.new_full((N + 1, M + 1), 1e9)
    D[0, 0] = 0.0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            D[i, j] = cost[i - 1, j - 1] + min(
                D[i - 1, j - 1].item(), D[i - 1, j].item(), D[i, j - 1].item()
            )
    path: list[tuple[int, int]] = []
    i, j = N, M
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        _, i, j = min(
            [
                (D[i - 1, j - 1].item(), i - 1, j - 1),
                (D[i - 1, j].item(), i - 1, j),
                (D[i, j - 1].item(), i, j - 1),
            ],
            key=lambda c: c[0],
        )
    return path[::-1]


def dtw_matched_pairs(
    pred_embeds: torch.Tensor, gt_embeds: torch.Tensor
) -> list[tuple[int, int]]:
    """One-to-one (pred word -> GT word) matches via hard DTW on cosine cost."""
    cost = 1.0 - F.cosine_similarity(
        pred_embeds.unsqueeze(1), gt_embeds.unsqueeze(0), dim=-1
    )
    matched: dict[int, int] = {}
    for pi, gi in hard_dtw_path(cost):
        matched.setdefault(pi, gi)
    return list(matched.items())


# --- Prediction CSV helpers (CER / WER / SemER) ----------------------------
def _encode_sentences(sentences: list[str], tok, mdl, batch_size: int = 64) -> np.ndarray:
    all_embs = []
    for i in range(0, len(sentences), batch_size):
        enc = tok(
            sentences[i : i + batch_size],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = mdl(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        all_embs.append(F.normalize(emb.float(), p=2, dim=-1).numpy())
    return np.concatenate(all_embs, axis=0)


def compute_semer_batch(preds: list[str], refs: list[str]) -> list[float]:
    """Semantic error rate per sample: L2 distance of mean-pooled RoBERTa embeddings."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ROBERTA_PATH)
    mdl = AutoModel.from_pretrained(ROBERTA_PATH)
    mdl.eval()
    emb_pred = _encode_sentences(preds, tok, mdl)
    emb_ref = _encode_sentences(refs, tok, mdl)
    return np.linalg.norm(emb_pred - emb_ref, axis=-1).tolist()


def prediction_fieldnames(
    has_ctc: bool = False, has_segment_meta: bool = False
) -> list[str]:
    cols: list[str] = []
    if has_segment_meta:
        cols += ["sentence_UID", "subject"]
    cols.append("true_text")
    if has_ctc:
        cols += ["ctc_text", "CTC_CER"]
    cols += ["pred_text", "CER", "WER", "SemER"]
    return cols


def compute_sample_metrics(
    true_texts: list[str],
    pred_texts: list[str],
    ctc_texts: list[str] | None = None,
    with_semer: bool = True,
) -> list[dict]:
    """Per-sentence CER / WER / SemER (and optional CTC CER) for the predictions CSV."""
    from torchmetrics.text import CharErrorRate, WordErrorRate

    cer_fn, wer_fn = CharErrorRate(), WordErrorRate()
    rows: list[dict] = []
    for i, (tgt, prd) in enumerate(zip(true_texts, pred_texts)):
        row: dict[str, tp.Any] = {
            "true_text": tgt,
            "pred_text": prd,
            "CER": cer_fn([prd], [tgt]).item(),
            "WER": wer_fn([prd], [tgt]).item(),
        }
        if ctc_texts:
            row["ctc_text"] = ctc_texts[i]
            row["CTC_CER"] = cer_fn([ctc_texts[i]], [tgt]).item()
        rows.append(row)
    if with_semer and true_texts:
        for row, val in zip(rows, compute_semer_batch(pred_texts, true_texts)):
            row["SemER"] = val
    return rows


# --- CTC label <-> text helpers --------------------------------------------
def label_to_text(ids: list[int]) -> str:
    """Map a CTC target id sequence to text ('&' -> space)."""
    chars = [letters_withblank[i] for i in ids if 0 < i < len(letters_withblank)]
    return "".join(" " if c == "&" else c for c in chars)


def ctc_greedy_decode(ctc_logits: torch.Tensor) -> list[str]:
    """Greedy CTC decode (blank=0, collapse repeats, '&' -> space)."""
    preds = ctc_logits.argmax(dim=-1)
    texts: list[str] = []
    for b in range(preds.shape[0]):
        chars: list[str] = []
        prev = 0
        for t in range(preds.shape[1]):
            c = preds[b, t].item()
            if c != prev and c != 0 and c < len(letters_withblank):
                ch = letters_withblank[c]
                chars.append(" " if ch == "&" else ch)
            prev = c
        texts.append("".join(chars))
    return texts


# --- Channel positions -----------------------------------------------------
class ChannelPositions2D(_ChannelPositions):
    """Re-enable 2D channel positions for MEG to match the paper."""

    def model_post_init(self, log__: tp.Any) -> None:
        BaseStatic.model_post_init(self, log__)
        if self.neuro is not None:
            if self.event_types not in {"MneRaw", self.neuro.event_types}:
                raise ValueError(
                    f"event_types={self.event_types} must match "
                    f"neuro.event_types={self.neuro.event_types}."
                )
            self._neuro = self.neuro


# --- Data / experiment helpers ---------------------------------------------
def accelerator(devices: int) -> tuple[str, int]:
    """Return (accelerator, n_devices), capped to the available GPUs."""
    if torch.cuda.is_available():
        return "gpu", max(1, min(devices, torch.cuda.device_count()))
    return "cpu", 1


def build_events(study, transforms, tail_range: tuple[float, float] = (0.4, 0.5)):
    """Run the study and its transforms, then extend each sentence window by a
    random tail (so a segment never ends exactly on the last keystroke)."""
    events = study.run()
    for transform in transforms:
        events = transform.run(events)
    events = ns.events.standardize_events(events)
    sentences = events[events.type == "Sentence"]
    events.loc[sentences.index, "duration"] = sentences.duration + np.random.uniform(
        tail_range[0], tail_range[1], len(sentences)
    )
    return events


def prepare_word_embeddings(data, word_extractor_config) -> dict[str, list]:
    """Per-sentence list of LLM word embeddings used as the contrastive target."""
    from neuralset.events import etypes

    events = build_events(data.study, data.transforms, (data.tail_min, data.tail_max))
    word_events = events[events["type"] == "Word"]
    assert len(word_events) > 0, "No Word events; add WordCreator to data.transforms."

    ext = ns.extractors.HuggingFaceText(**word_extractor_config)
    ext.prepare(events)
    word_events = word_events.sort_values(["sentence_UID", "word_order"])

    lookup: dict[str, list] = {}
    for sent_text, grp in word_events.groupby("sentence", sort=False):
        if sent_text in lookup:
            continue
        word_ev_list = [
            etypes.Word(
                text=row["text"],
                context=row["context"],
                sentence=row["sentence"],
                start=float(row["start"]),
                duration=float(row["duration"]),
                timeline=str(row["timeline"]),
            )
            for _, row in grp.iterrows()
        ]
        lookup[str(sent_text)] = list(ext._get_data(word_ev_list))
    return lookup
