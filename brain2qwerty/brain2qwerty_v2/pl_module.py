# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchmetrics import Metric

from .augmentations import Preprocess, PreprocessConfig
from .ctc_segmenter import CTCSpaceSegmenter, build_intra_word_pooler
from .losses import CtcLoss, WordContrastiveLoss
from .metrics import CharacterErrorRate
from .utils import compute_output_lens, ctc_greedy_decode

log = logging.getLogger(__name__)


class NeuroLLMModule(pl.LightningModule):
    """End-to-end Brain2Qwerty V2 module: CTC + word-contrastive + LLM losses.

    One encoder forward feeds three jointly-trained objectives, combined as
    ``loss = w_ctc * CTC + w_con * contrastive + w_llm * LLM`` where the weights
    come from ``(1-alpha-beta, alpha, beta)``, gated by epoch (``ctc/contrastive/
    llm_start_epoch``) and renormalised over the currently active losses. The CTC
    head segments encoder frames into pseudo-words (contrastive target) and seeds a
    LoRA-adapted LLM that autoregressively generates the sentence.
    """

    def __init__(
        self,
        *,
        network: nn.Module,
        llm: nn.Module,
        tokenizer,
        word_proj_adapter: nn.Module,
        word_embed_lookup: dict[str, list[np.ndarray]] | None = None,
        word_pool_dim: int = 1024,
        word_pool_n_layers: int = 2,
        seg_include_blanks: bool = True,
        alpha: float = 0.1,
        beta: float = 0.01,
        loss_alpha: float = 0.7,
        ctc_start_epoch: int = 0,
        contrastive_start_epoch: int = 0,
        llm_start_epoch: int = 0,
        sys_prompt: str = "CTC: ",
        mid_prompt: str = "\nMEG: ",
        resp_prompt: str = "\nOutput: ",
        max_new_tokens: int = 60,
        num_beams: int = 16,
        val_num_beams: int = 1,
        length_penalty: float = 0.2,
        label_smoothing: float = 0.02,
        meg_dropout_rate: float = 0.1,
        ctc_dropout_rate: float = 0.1,
        optimizer_config: dict | None = None,
        scheduler_config: dict | None = None,
        preprocess_config: dict | None = None,
        encoder_lr: float | None = None,
        llm_metrics: dict[str, Metric] | None = None,
        save_dir: str | None = None,
    ):
        super().__init__()
        self.network = network
        self.llm = llm
        self.tokenizer = tokenizer
        self.word_proj_adapter = word_proj_adapter

        self.optimizer_config = optimizer_config or {}
        self.scheduler_config = scheduler_config or {}
        self.preprocess = Preprocess(
            **PreprocessConfig(**(preprocess_config or {})).model_dump()
        )

        self.ctc_loss = CtcLoss()
        self.loss_alpha = loss_alpha
        self.val_ctc_cer = CharacterErrorRate()
        self.test_ctc_cer = CharacterErrorRate()

        self.word_segmenter = CTCSpaceSegmenter(
            include_blanks=seg_include_blanks,
            min_word_frames=1,
            intra_word_pooler=build_intra_word_pooler(word_pool_dim, word_pool_n_layers),
        )
        self.word_contrastive_loss = WordContrastiveLoss()
        self.word_embed_lookup = word_embed_lookup or {}
        self._save_dir = save_dir

        assert alpha + beta < 1.0, f"alpha + beta must be < 1, got {alpha + beta}"
        self.ctc_weight = 1.0 - alpha - beta
        self.contrastive_weight = alpha
        self.llm_weight = beta
        self.ctc_start_epoch = ctc_start_epoch
        self.contrastive_start_epoch = contrastive_start_epoch
        self.llm_start_epoch = llm_start_epoch
        self.encoder_lr = encoder_lr

        self.sys_prompt = sys_prompt
        self.mid_prompt = mid_prompt
        self.resp_prompt = resp_prompt
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.val_num_beams = val_num_beams
        self.length_penalty = length_penalty
        self.label_smoothing = label_smoothing
        self.meg_dropout_rate = meg_dropout_rate
        self.ctc_dropout_rate = ctc_dropout_rate

        llm_metrics = llm_metrics or {}
        modules: dict[str, Metric] = {}
        for name, m in llm_metrics.items():
            for split in ("val", "test"):
                modules[f"llm_{split}_{name}"] = m.clone()
        self._llm_metrics = nn.ModuleDict(modules)
        self._llm_metric_names = list(llm_metrics.keys())

        self._test_predictions: list[dict] = []

    # --- loss gating ---------------------------------------------------
    def _contrastive_active(self, step_name: str) -> bool:
        if self.contrastive_weight <= 0.0:
            return False
        return step_name == "test" or self.current_epoch >= self.contrastive_start_epoch

    def _llm_active(self, step_name: str) -> bool:
        if self.llm_weight <= 0.0:
            return False
        return step_name == "test" or self.current_epoch >= self.llm_start_epoch

    def _lookup_text_embeds(self, sentences: list[str]) -> list[torch.Tensor] | None:
        result = []
        for sent in sentences:
            if sent not in self.word_embed_lookup:
                return None
            arrays = self.word_embed_lookup[sent]
            result.append(torch.from_numpy(np.stack(arrays)).float().to(self.device))
        return result

    # --- LLM helpers ---------------------------------------------------
    def _tok_embed(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        base = self.llm.get_base_model()
        if not ids:
            return torch.zeros(
                0, base.config.hidden_size, device=self.device, dtype=base.dtype
            )
        return base.model.embed_tokens(torch.tensor(ids, device=self.device))

    def _pad_word_embeds(self, word_embeds_list):
        B = len(word_embeds_list)
        max_len = max((w.shape[0] for w in word_embeds_list), default=0)
        if max_len == 0:
            D = self.llm.get_base_model().config.hidden_size
            return (
                torch.zeros(B, 1, D, device=self.device),
                torch.zeros(B, 1, device=self.device),
            )
        D = word_embeds_list[0].shape[-1]
        dev = word_embeds_list[0].device
        padded = torch.zeros(B, max_len, D, device=dev)
        mask = torch.zeros(B, max_len, device=dev)
        for i, w in enumerate(word_embeds_list):
            if w.shape[0] > 0:
                padded[i, : w.shape[0]] = w
                mask[i, : w.shape[0]] = 1.0
        return padded, mask

    def _build_llm_prefixes(self, adapted_embeds, neuro_mask, ctc_texts, B):
        sys_emb = self._tok_embed(self.sys_prompt)
        resp_emb = self._tok_embed(self.resp_prompt)
        mid_emb = self._tok_embed(self.mid_prompt)
        prefixes: list[torch.Tensor] = []
        for i in range(B):
            n_valid = int(neuro_mask[i].sum().item())
            words_i = adapted_embeds[i, :n_valid]
            ctc_emb_i = self._tok_embed(ctc_texts[i])
            if self.training:
                if self.meg_dropout_rate > 0 and words_i.numel() > 0:
                    keep = (
                        torch.rand(words_i.shape[0], 1, device=words_i.device)
                        > self.meg_dropout_rate
                    )
                    words_i = words_i * keep
                if self.ctc_dropout_rate > 0 and ctc_emb_i.numel() > 0:
                    keep = (
                        torch.rand(ctc_emb_i.shape[0], 1, device=ctc_emb_i.device)
                        > self.ctc_dropout_rate
                    )
                    ctc_emb_i = ctc_emb_i * keep
            parts = [sys_emb, ctc_emb_i]
            if words_i.shape[0] > 0:
                parts += [mid_emb, words_i]
            parts.append(resp_emb)
            prefixes.append(torch.cat([p for p in parts if p.numel() > 0], dim=0))
        return prefixes

    def _compute_llm_loss(self, adapted_words, neuro_mask, ctc_texts, sentences):
        B = len(sentences)
        base = self.llm.get_base_model()
        targets = self.tokenizer(
            [s + self.tokenizer.eos_token for s in sentences],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(self.device)
        target_emb = base.model.embed_tokens(targets.input_ids)
        prefixes = self._build_llm_prefixes(adapted_words, neuro_mask, ctc_texts, B)

        sequences, all_labels = [], []
        for i in range(B):
            valid_tgt = targets.attention_mask[i].bool()
            tgt_ids_i = targets.input_ids[i, valid_tgt]
            tgt_emb_i = target_emb[i, valid_tgt]
            sequences.append(torch.cat([prefixes[i], tgt_emb_i], dim=0))
            all_labels.append(
                torch.cat(
                    [
                        torch.full(
                            (prefixes[i].shape[0],),
                            -100,
                            device=self.device,
                            dtype=torch.long,
                        ),
                        tgt_ids_i,
                    ]
                )
            )

        max_seq = max(s.shape[0] for s in sequences)
        D = sequences[0].shape[-1]
        input_embeds = sequences[0].new_zeros(B, max_seq, D)
        labels = torch.full((B, max_seq), -100, device=self.device, dtype=torch.long)
        attention_mask = torch.zeros(B, max_seq, device=self.device)
        for i in range(B):
            L = sequences[i].shape[0]
            input_embeds[i, :L] = sequences[i]
            labels[i, :L] = all_labels[i]
            attention_mask[i, :L] = 1.0
        input_embeds = input_embeds.to(base.dtype)

        outputs = self.llm(inputs_embeds=input_embeds, attention_mask=attention_mask)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            label_smoothing=self.label_smoothing,
        )

    @torch.no_grad()
    def _generate_text(self, adapted_words, neuro_mask, ctc_texts, step_name="test"):
        B = adapted_words.shape[0]
        base = self.llm.get_base_model()
        prefixes = self._build_llm_prefixes(adapted_words, neuro_mask, ctc_texts, B)
        max_len = max(p.shape[0] for p in prefixes)
        D = prefixes[0].shape[-1]
        prefix = adapted_words.new_zeros(B, max_len, D)
        prefix_mask = adapted_words.new_zeros(B, max_len)
        for i, p in enumerate(prefixes):  # left-pad for generation
            L = p.shape[0]
            prefix[i, max_len - L :] = p
            prefix_mask[i, max_len - L :] = 1.0
        prefix = prefix.to(base.dtype)

        beams = self.num_beams if step_name == "test" else self.val_num_beams
        gen_kwargs: dict = dict(
            inputs_embeds=prefix,
            attention_mask=prefix_mask,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
        )
        if beams > 1:
            gen_kwargs["num_beams"] = beams
        if self.length_penalty != 1.0:
            gen_kwargs["length_penalty"] = self.length_penalty
            gen_kwargs["early_stopping"] = True
        generated_ids = self.llm.generate(**gen_kwargs)
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # --- core step -----------------------------------------------------
    def _run_step(self, batch, batch_idx, step_name):
        data = batch.data
        if step_name == "train":
            data = self.preprocess(data)

        B = data["neuros"].shape[0]
        model_out = self.network(data["neuros"], data["days"], data.get("chan_pos"))
        z_final = model_out["z_final"]
        ctc_logits = model_out["c_out"]

        out_lens = compute_output_lens(self.network, data["neuro_sizes"])
        out_lens = torch.clamp(out_lens, min=0, max=ctc_logits.shape[1])

        if ctc_logits.shape[1] < data["phoneme_sizes"].max():
            log.warning(
                "[%s] batch %d skipped: pred_len < target_len", step_name, batch_idx
            )
            return (ctc_logits * 0).sum(), ctc_logits, data["phonemes"]

        ctc_on = step_name == "test" or self.current_epoch >= self.ctc_start_epoch
        con_on = self._contrastive_active(step_name)
        llm_on = self._llm_active(step_name)

        w_ctc = self.ctc_weight if ctc_on else 0.0
        w_con = self.contrastive_weight if con_on else 0.0
        w_llm = self.llm_weight if llm_on else 0.0
        w_sum = w_ctc + w_con + w_llm
        if w_sum > 0:
            w_ctc, w_con, w_llm = w_ctc / w_sum, w_con / w_sum, w_llm / w_sum

        losses: list[torch.Tensor] = []

        # CTC (auxiliary head logits blended with the final logits)
        if ctc_on:
            loss_z = self.ctc_loss(
                model_out["z"], data["phonemes"], out_lens, data["phoneme_sizes"]
            )
            loss_c = self.ctc_loss(
                ctc_logits, data["phonemes"], out_lens, data["phoneme_sizes"]
            )
            ctc_loss = (1 - self.loss_alpha) * loss_c + self.loss_alpha * loss_z
            losses.append(w_ctc * ctc_loss)
            self.log(
                f"{step_name}/loss_ctc",
                ctc_loss,
                on_step=(step_name == "train"),
                on_epoch=True,
                prog_bar=True,
                batch_size=B,
            )

        if step_name != "train":
            metric = self.val_ctc_cer if step_name == "val" else self.test_ctc_cer
            metric.update(ctc_logits, data["phonemes"], out_lens, data["phoneme_sizes"])

        # Word segmentation (shared by contrastive + LLM)
        sentences = [seg.trigger.text for seg in batch.segments]
        word_embeds_list = None
        if (con_on or llm_on) and sentences:
            word_embeds_list = self.word_segmenter(z_final, ctc_logits)
            # Map encoder word embeddings into the LLM word-embedding space, the
            # space the contrastive target lives in (Identity when the encoder
            # dim already equals the LLM hidden dim).
            word_embeds_list = [self.word_proj_adapter(w) for w in word_embeds_list]

        # Word-level contrastive alignment
        if con_on and word_embeds_list is not None:
            text_embeds = self._lookup_text_embeds(sentences)
            if text_embeds is not None:
                c_loss = self.word_contrastive_loss(word_embeds_list, text_embeds)["loss"]
                losses.append(w_con * c_loss)
                self.log(
                    f"{step_name}/loss_contrastive",
                    c_loss,
                    on_step=(step_name == "train"),
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=B,
                )

        # LLM generation
        if llm_on and word_embeds_list is not None and sentences:
            padded_words, neuro_mask = self._pad_word_embeds(word_embeds_list)
            ctc_texts = ctc_greedy_decode(ctc_logits.detach())
            llm_loss = self._compute_llm_loss(
                padded_words, neuro_mask, ctc_texts, sentences
            )
            losses.append(w_llm * llm_loss)
            self.log(
                f"{step_name}/loss_llm",
                llm_loss,
                on_step=(step_name == "train"),
                on_epoch=True,
                prog_bar=True,
                batch_size=B,
            )

            if step_name != "train":
                pred_texts = self._generate_text(
                    padded_words, neuro_mask, ctc_texts, step_name
                )
                for name in self._llm_metric_names:
                    # SemER loads RoBERTa on every rank; only score it at test to
                    # keep validation epochs fast and memory-light.
                    if name == "SemER" and step_name != "test":
                        continue
                    key = f"llm_{step_name}_{name}"
                    self._llm_metrics[key].update(pred_texts, sentences)
                    self.log(
                        f"{step_name}/{name}",
                        self._llm_metrics[key],
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        batch_size=B,
                    )
                # Per-sentence rows are only saved to CSV at test time.
                if step_name == "test":
                    for i in range(B):
                        seg = batch.segments[i]
                        self._test_predictions.append(
                            {
                                "true_text": sentences[i],
                                "ctc_text": ctc_texts[i],
                                "pred_text": pred_texts[i],
                                "subject": seg.trigger.extra.get("subject", ""),
                                "sentence_UID": seg.trigger.extra.get("sentence_UID", ""),
                            }
                        )

        total_loss = sum(losses) if losses else (ctc_logits * 0).sum()
        self.log(
            f"{step_name}/loss",
            total_loss,
            on_step=(step_name == "train"),
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )
        return total_loss, ctc_logits, data["phonemes"]

    def training_step(self, batch, batch_idx):
        return self._run_step(batch, batch_idx, "train")[0]

    def validation_step(self, batch, batch_idx):
        _, y_pred, y_true = self._run_step(batch, batch_idx, "val")
        return y_pred, y_true

    def test_step(self, batch, batch_idx):
        _, y_pred, y_true = self._run_step(batch, batch_idx, "test")
        return y_pred, y_true

    def on_validation_epoch_end(self) -> None:
        self.log("val/cer_epo", self.val_ctc_cer.compute(), prog_bar=True)
        self.val_ctc_cer.reset()
        # Before the LLM phase, val/WER is not produced yet; log a sentinel so the
        # best_llm checkpoint (monitor=val/WER) always finds its key. Once the LLM
        # phase starts, the real (lower) WER replaces it as the checkpoint's best.
        if self.current_epoch < self.llm_start_epoch:
            self.log("val/WER", 1.0)

    def on_test_epoch_end(self) -> None:
        self.log("test/cer_epo", self.test_ctc_cer.compute(), prog_bar=True)
        self.test_ctc_cer.reset()
        self._test_predictions = []

    # --- optimiser -----------------------------------------------------
    def configure_optimizers(self):
        encoder_ids = {id(p) for p in self.network.parameters()}
        enc_trainable = [p for p in self.network.parameters() if p.requires_grad]
        other_trainable = [
            p for p in self.parameters() if id(p) not in encoder_ids and p.requires_grad
        ]
        base_lr = self.optimizer_config.get("lr", 4e-4)
        enc_lr = self.encoder_lr or base_lr
        wd = self.optimizer_config.get("weight_decay", 1e-3)

        groups = []
        if enc_trainable:
            groups.append({"params": enc_trainable, "lr": enc_lr})
        if other_trainable:
            groups.append({"params": other_trainable, "lr": base_lr})
        optimizer = optim.AdamW(
            groups or [{"params": list(self.parameters())}], lr=base_lr, weight_decay=wd
        )

        # Warmup (linear) followed by cosine decay, as in the paper.
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.scheduler_config.get("warmup_steps", 500)
        cosine_steps = max(total_steps - warmup_steps, 1)
        warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.scheduler_config.get("T_max", cosine_steps),
            eta_min=self.scheduler_config.get("eta_min", 1e-6),
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": self.trainer.accumulate_grad_batches or 1,
            },
        }
