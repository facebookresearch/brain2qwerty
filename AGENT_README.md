<!--
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
-->

# Agent guide

A deep orientation for an AI coding agent working in this
repository. Read this once before making changes; it captures how the code is
organised, how data flows, and what is faithful to the papers.

---

## 1. What this repo is

Two generations of **Brain2Qwerty**, a system that decodes typed sentences from
non-invasive MEG. Each generation is a **self-contained Python package**:

- **`brain2qwerty_v1/`** — keystroke-level decoder (Conv encoder + sentence-level
  Transformer), trained with per-keystroke cross-entropy. Dataset: **SpanishBCBL**
  (`Pinet2024Meg`). Reported MEG CER ≈ 0.38 (Conv+Transformer, no language model);
  the optional N-gram rescoring lowers it further.
- **`brain2qwerty_v2/`** — sentence-level end-to-end decoder. A Conv+Conformer CTC
  encoder, a word-level contrastive aligner (SigLIP + DTW), and a LoRA-adapted LLM,
  trained jointly on a **staged 3-loss schedule**. Dataset:
  **EnglishBCBL** (`PinetAudio2025`, not yet public).

The two packages are intentionally parallel: same file names, same config/CLI
conventions. Learn one and you know the other.

---

## 2. Layout

```
brain2qwerty/
├── brain2qwerty_v1/        keystroke decoder
│   ├── config/
│   │   ├── model_config.py   architecture dicts (ENCODER, TRANSFORMER)
│   │   └── xp_config.py      experiment_config() + debug_config()
│   ├── cli.py                shared --wandb argument helpers
│   ├── main.py               Data + Experiment + the {debug,train,eval,cache} CLI
│   ├── transforms.py         SpanishBCBLPreprocessing, Brain2QwertyV1Splitter
│   ├── models.py             (V1 has no model code beyond configs; see utils)
│   ├── utils.py              BUTTON_MAPPING, CHAR_INDEX, ChannelPositions2D, DDP sampler
│   ├── pl_module.py          BrainModule (LightningModule)
│   ├── metrics.py            CER
│   ├── callbacks.py          LogSentencePredictions
│   ├── scripts/              extract_predictions.py, ngram_decoding.py
│   └── README.md
├── brain2qwerty_v2/        sentence decoder (CTC + contrastive + LLM)
│   ├── config/{model_config,xp_config}.py
│   ├── cli.py, main.py       (main = SentenceDataset config glue + Experiment)
│   ├── data.py               SentenceDataset (jitter + padded collate)
│   ├── transforms.py         EnglishBCBLPreprocessing, Brain2QwertyV2Splitter,
│   │                          WordCreator, SentenceKeySeq (CTC-label extractor)
│   ├── models.py             ChannelPositions2D, ConvConformer(+Model),
│   │                          ctc_greedy_decode, label_to_text, CTCSpaceSegmenter
│   ├── losses.py             CtcLoss, WordContrastiveLoss
│   ├── metrics.py            CharacterErrorRate (CTC monitor), SemER
│   ├── augmentations.py      Preprocess (on-device MEG augmentation)
│   ├── pl_module.py          NeuroLLMModule (3-loss, staged schedule)
│   ├── utils.py              key_to_int, apply_jitter, DTW, MLP, prediction helpers
│   └── README.md
├── studies/                 vendored study definitions
│   └── spanishbcbl.py        Pinet2024Meg  (alias "SpanishBCBL")
├── tests/                   fast CPU unit tests + opt-in live data checks
├── pyproject.toml           pinned dependencies (reproducibility)
└── README.md, AGENT_README.md
```

---

## 3. Dependencies and the public-API contract

Everything runs on the **public PyPI** releases of `neuralset` and `neuraltrain`
(part of [neuroai](https://github.com/facebookresearch/neuroai)), pinned exactly in
`pyproject.toml`. Do **not** vendor or fork them.

- **Data layer = `neuralset`**: `Study(name=...).run()` → a pandas events frame →
  `EventsTransform.run(events)` (chained) → `standardize_events` →
  `list_segments(...)` → `SegmentDataset(extractors=..., segments=...)` → a `Batch`
  (with `.data` tensors and `.segments` carrying the trigger events).
- **Models / losses / optimizers = `neuraltrain`**: discriminated config classes
  (`BaseModelConfig`, `BaseLoss`, `LightningOptimizer`, `WandbLoggerConfig`).
- **Caching / scheduling = `exca`** (`MapInfra`, `cluster="auto"` to fan out over
  SLURM). Owned transitively by neuralset.

Two model behaviours are **not** in the public packages and are re-added locally:

1. `ChannelPositions2D` (in each package's `utils.py`/`models.py`): the public
   `ChannelPositions` forbids 2D MEG layouts, but the channel merger needs the 2D
   Fourier positional embedding from the paper. The subclass bypasses only that
   guard; numerics are identical.
2. `ConvConformer` / `ConvConformerModel` (V2 `models.py`): subclass the public
   `ConvTransformer` to re-add the auxiliary CTC head (`aux_prediction`) and the
   per-frame `z_final` output used by the word segmenter.

Study definitions are **vendored** under `studies/`. Importing `studies` registers
them so `Study(name="Pinet2024Meg")` / `"PinetAudio2025"` resolve. Every entry point
does `import studies` for this side effect.

---

## 4. How a run is wired

`main.py` exposes the experiment in four modes (same set for both packages); each
builds the config from `config/xp_config` and runs the pydantic `Experiment`:

- `python -m <pkg>.main debug` → `debug_config()`: 1 timeline, 2 epochs, 1 GPU, no checkpoints.
- `python -m <pkg>.main train` → `experiment_config()`; 8 GPUs by default, auto-capped to available.
- `python -m <pkg>.main eval --ckpt` → `eval_only=True`, loads the checkpoint and runs the
  test split on a **single device** so the prediction callback captures the whole split.
- `python -m <pkg>.main cache` → builds the dataloaders once to pre-warm the feature
  cache (`--debug` for the 1-timeline subset).
- `--wandb` → attaches a `WandbLoggerConfig`; the host is read from the `WANDB_HOST`
  env var (never hardcoded). Without `--wandb`, a `CSVLogger` still records metrics so
  runs are legible.

`Experiment.run()` → `data.build()` (dataloaders) → build model(s) → **materialise
lazy params with a dummy forward** → `Trainer.fit` → `Trainer.test`. Multi-GPU uses
`DDPStrategy(find_unused_parameters=True)`.

### V1 forward (BrainModule)
encoder produces one embedding per keystroke window → embeddings are **grouped by
`sentence_UID`** (via `seg.trigger.extra`) → sentence-level Transformer → linear head
→ per-keystroke cross-entropy. CER is the Levenshtein-based `CER` metric. The
`SentenceGroupedDistributedSampler` keeps a sentence's keystrokes on one rank.

### V2 forward (NeuroLLMModule)
one encoder forward feeds three losses, combined as
`loss = w_ctc·CTC + w_con·contrastive + w_llm·LLM`, with weights `(1-α-β, α, β)`
gated by epoch (`ctc/contrastive/llm_start_epoch`) and renormalised over the active
losses. The CTC head has an auxiliary logits path blended via `loss_alpha`
(`0.3·c_out + 0.7·z`). `CTCSpaceSegmenter` groups encoder frames into pseudo-words
(contrastive target = LLM word embeddings); the LoRA LLM autoregressively generates
the sentence from `[CTC text] + [MEG word embeds]`. Reported test metrics: **CER, WER,
SemER** (RoBERTa-large embedding distance); `val/cer_epo` is the CTC-greedy monitor
used for checkpointing.

---

## 5. Configuration & environment

The configs are the **full, explicit** values used in the papers — treat them as the
source of truth for reproduction. `pyproject.toml` exact-pins every direct
dependency; `requirements.lock` additionally pins the full transitive closure to
the exact versions used (`pip install -r requirements.lock`). Environment
variables:

| Var | Meaning |
|----|----|
| `BRAIN2QWERTY_STUDIES` | studies root for V1 (SpanishBCBL) |
| `BRAIN2QWERTY_STUDIES_EN` | EnglishBCBL data root for V2 (`.../pinet2025`) |
| `BRAIN2QWERTY_CACHE` | exca feature/timeline cache (large; persistent) |
| `BRAIN2QWERTY_RESULTS` | checkpoints + prediction CSVs + CSV metric logs |
| `BRAIN2QWERTY_ROBERTA` | RoBERTa id/path for SemER (default `roberta-large`) |
| `WANDB_HOST` | optional W&B base URL for `--wandb` |
| `HF_HOME` | HuggingFace cache (LLM, RoBERTa); pre-download for offline nodes |

---

## 6. Running

```bash
python -m brain2qwerty_v1.main cache                # pre-warm cache (--debug for subset)
python -m brain2qwerty_v1.main debug                # V1 smoke test (1 GPU)
python -m brain2qwerty_v2.main debug                # V2 smoke test (1 GPU)
python -m brain2qwerty_v1.main train --wandb        # full training (1 node, 8 GPUs)
python -m brain2qwerty_v1.main eval  --ckpt best.ckpt
python -m brain2qwerty_v1.scripts.extract_predictions --input <results>/callbacks
pytest tests -q                                     # fast CPU unit tests
```

**Multi-node SLURM**: launch with `srun` and `--ntasks-per-node = --gpus-per-node`
(one rank per GPU); Lightning's `SLURMEnvironment` is auto-detected.

**Pre-warm the cache first** for full runs (feature extraction is CPU-bound). Build
`Data` with `neuro.infra.cluster="auto"` and call `data.build()` once — exca fans the
per-recording MEG extraction out to up to `max_jobs` SLURM jobs and caches it; the GPU
training job then starts against a warm cache (set the same `cluster=None` + folder).

---

## 7. Conventions

- Only the **final selected model** is implemented — one scheduler, one loss path,
  no ablation switches.
- Keep changes **faithful to the published models**; verify a `debug` run still trains
  (losses decrease, no NaN) before and after any change.
- Files start straight at imports — no leading "purpose of this script" comments;
  document **classes** with concise docstrings instead.
- `git add` only intended files; the repo's pre-commit runs black/isort.
