from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from brain2qwerty_v2.pl_module import NeuroLLMModule


class _TokBatch:
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def to(self, _device):
        return self


class _Tokenizer:
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 32 + 1 for c in text]

    def __call__(self, texts, return_tensors="pt", padding=True, add_special_tokens=False):
        max_len = max(len(t) for t in texts)
        ids = torch.zeros(len(texts), max_len, dtype=torch.long)
        mask = torch.zeros(len(texts), max_len, dtype=torch.long)
        for i, t in enumerate(texts):
            vals = torch.tensor([ord(c) % 32 + 1 for c in t], dtype=torch.long)
            ids[i, : len(vals)] = vals
            mask[i, : len(vals)] = 1
        return _TokBatch(ids, mask)

    def batch_decode(self, generated_ids, skip_special_tokens=True):
        return ["decoded"] * generated_ids.shape[0]


class _DummyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.dtype = torch.float32
        self.model = SimpleNamespace(embed_tokens=nn.Embedding(128, 8))


class _DummyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = _DummyBase()
        self.head = nn.Linear(8, 8)

    def get_base_model(self):
        return self.base

    def forward(self, inputs_embeds, attention_mask):
        b, t, _d = inputs_embeds.shape
        return SimpleNamespace(logits=torch.zeros(b, t, 16, device=inputs_embeds.device))

    def generate(self, **kwargs):
        b = kwargs["inputs_embeds"].shape[0]
        return torch.ones(b, 2, dtype=torch.long, device=kwargs["inputs_embeds"].device)


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, neuros, days, chan_pos=None):
        b, t, _c = neuros.shape
        z = torch.randn(b, t, 8, device=neuros.device)
        return {"z_final": z, "c_out": torch.randn(b, t, 29, device=neuros.device), "z": torch.randn(b, t, 29, device=neuros.device)}


def _build_module(alpha=0.1, beta=0.01, contrastive_start_epoch=2, llm_start_epoch=3):
    return NeuroLLMModule(
        network=_TinyNet(),
        llm=_DummyLLM(),
        tokenizer=_Tokenizer(),
        word_proj_adapter=nn.Linear(8, 8),
        word_embed_lookup={"hi": [np.zeros(8), np.ones(8)]},
        word_pool_dim=8,
        word_pool_n_layers=1,
        alpha=alpha,
        beta=beta,
        contrastive_start_epoch=contrastive_start_epoch,
        llm_start_epoch=llm_start_epoch,
        optimizer_config={"lr": 1e-3, "weight_decay": 0.0},
        scheduler_config={"warmup_steps": 1, "T_max": 4, "eta_min": 1e-5},
    )


def test_activity_gating_and_lookup_text_embeds():
    m = _build_module()
    m._trainer = SimpleNamespace(current_epoch=0)

    assert m._contrastive_active("val") is False
    assert m._llm_active("val") is False
    assert m._contrastive_active("test") is True
    assert m._llm_active("test") is True

    embeds = m._lookup_text_embeds(["hi"])
    assert embeds is not None and embeds[0].shape == (2, 8)
    assert m._lookup_text_embeds(["missing"]) is None


def test_pad_word_embeds_and_tok_embed_shapes():
    m = _build_module()

    z, mask = m._pad_word_embeds([torch.randn(2, 8), torch.randn(1, 8)])
    assert z.shape == (2, 2, 8)
    assert mask.shape == (2, 2)

    empty_z, empty_mask = m._pad_word_embeds([torch.zeros(0, 8)])
    assert empty_z.shape[0] == 1
    assert empty_mask.shape[0] == 1

    tok = m._tok_embed("abc")
    assert tok.shape[1] == 8


def test_validation_end_and_configure_optimizers():
    m = _build_module()
    m._trainer = SimpleNamespace(current_epoch=0, estimated_stepping_batches=6, accumulate_grad_batches=2)

    logs = []
    m.log = lambda name, value, **kwargs: logs.append(name)
    m.on_validation_epoch_end()
    assert "val/cer_epo" in logs
    assert "val/WER" in logs

    opt_cfg = m.configure_optimizers()
    assert "optimizer" in opt_cfg and "lr_scheduler" in opt_cfg
    assert len(opt_cfg["optimizer"].param_groups) >= 1
