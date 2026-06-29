# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from edit_distance import SequenceMatcher
from torchmetrics import Metric

from .utils import ROBERTA_PATH


class CharacterErrorRate(Metric):
    """CTC greedy character error rate on the encoder logits (blank=0, collapse repeats).

    Monitors the CTC head during training/validation (used for checkpoint
    selection); distinct from the LLM-output CER/WER/SemER reported at test time.
    """

    def __init__(self) -> None:
        super().__init__()
        for name in ("total_edit_distance", "total_length"):
            self.add_state(name, default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_pred, y_true, adjusted_x_len, y_len):
        for idx in range(y_pred.shape[0]):
            decoded = torch.argmax(y_pred[idx, : adjusted_x_len[idx], :], dim=-1)
            decoded = torch.unique_consecutive(decoded, dim=-1)
            decoded = [x.item() for x in decoded if x.item() != 0]
            true_seq = y_true[idx, : y_len[idx]]
            self.total_edit_distance += SequenceMatcher(
                a=true_seq.tolist(), b=decoded
            ).distance()
            self.total_length += len(true_seq)

    def compute(self):
        if self.total_length == 0:
            return torch.tensor(0.0)
        return self.total_edit_distance.float() / self.total_length


class SemanticErrorRate(Metric):
    """Semantic error rate: L2 distance between L2-normalised, mean-pooled
    RoBERTa-large embeddings of the predicted and reference sentences.

    Note: this runs a RoBERTa-large forward pass per sentence and is slow, so it
    is computed at test time only (not every validation epoch).
    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(
        self, model_name_or_path: str | None = None, batch_size: int = 32, **kwargs
    ):
        super().__init__(**kwargs)
        self.model_path = model_name_or_path or ROBERTA_PATH
        self.batch_size = batch_size
        self._encoder: tuple | None = None
        self.add_state("error_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    def _get_encoder(self):
        if self._encoder is None:
            from transformers import AutoModel, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.model_path)
            mdl = AutoModel.from_pretrained(self.model_path)
            mdl.eval()
            self._encoder = (tok, mdl)
        return self._encoder

    @torch.no_grad()
    def _encode(self, sentences: list[str]) -> torch.Tensor:
        tok, mdl = self._get_encoder()
        mdl = mdl.to(self.device)
        all_embs: list[torch.Tensor] = []
        for i in range(0, len(sentences), self.batch_size):
            enc = tok(
                sentences[i : i + self.batch_size],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)
            out = mdl(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_embs.append(F.normalize(emb.float(), p=2, dim=-1))
        return torch.cat(all_embs, dim=0)

    def update(self, preds: list[str], target: list[str]) -> None:
        emb_pred = self._encode([str(p) for p in preds])
        emb_ref = self._encode([str(t) for t in target])
        self.error_sum += (emb_pred - emb_ref).norm(dim=-1).sum()
        self.count += len(preds)

    def compute(self) -> torch.Tensor:
        return self.error_sum / self.count.clamp(min=1)
