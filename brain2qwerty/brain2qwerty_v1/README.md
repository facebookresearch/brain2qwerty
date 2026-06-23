<!--
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
-->

# Brain2Qwerty V1

Official implementation of [**Non-invasive decoding of typed sentences from human brain activity**](https://arxiv.org/abs/2502.17480) (*Nature Neuroscience*, 2025).

Brain2Qwerty V1 uses a convolutional module that encodes keystroke-aligned MEG windows and a transformer that refines the predictions at the sentence level.

<p align="center">
  <img src="resources/approach.png" alt="Brain2Qwerty architecture: a convolutional module encodes 500 ms MEG windows around each keystroke and a transformer refines predictions at the sentence level." width="100%">
</p>

## This folder contains

- The keystroke decoding pipeline (Conv + Transformer) with training and evaluation using PyTorch Lightning
- Post-training scripts for [prediction extraction](scripts/extract_predictions.py) and [N-gram beam-search decoding](scripts/ngram_decoding.py)

## Installation

**Requirements:** Python 3.10+, CUDA-capable GPU.

```bash
pip install brain2qwerty                # core (neuralset / neuraltrain from PyPI)
pip install "brain2qwerty[lm]"          # adds KenLM for the N-gram decoding step
```

To use the exact dependency versions (every transitive package pinned), install
from the lockfile instead:

```bash
pip install -r requirements.lock
```

## Data

The SpanishBCBL dataset is hosted on the Hugging Face Hub by Meta:

- **Dataset:** `https://huggingface.co/datasets/<TBD>` *(link to be added)*

Download it (e.g. with `huggingface-cli download <TBD> --repo-type dataset --local-dir <path>`) and point the pipeline at it via environment variables:

```bash
export BRAIN2QWERTY_STUDIES="$HOME/brain2qwerty_data/studies"   # downloaded MEG recordings
export BRAIN2QWERTY_CACHE="$HOME/.cache/brain2qwerty"           # preprocessed features
export BRAIN2QWERTY_RESULTS="$HOME/.cache/brain2qwerty/results" # checkpoints / outputs
```

The preprocessed feature cache is created automatically on the first run.

## Quickstart

Each step is its own command. Training uses one node (8 GPUs by default) and automatically falls back to a single GPU.

```bash
# (optional) pre-warm the feature cache (--debug for the 1-timeline subset)
python -m brain2qwerty_v1.main cache

# short single-timeline run on 1 GPU (sanity check: loss decreases, no NaN)
python -m brain2qwerty_v1.main debug

# full training
python -m brain2qwerty_v1.main train

# evaluate a checkpoint on the test split
python -m brain2qwerty_v1.main eval --ckpt $BRAIN2QWERTY_RESULTS/best.ckpt
```

The full configuration lives in [`config/xp_config.py`](config/xp_config.py) (experiment) and [`config/model_config.py`](config/model_config.py) (architecture).

## Result extraction and analysis

The typical end-to-end workflow, from a trained model to the final (optionally
LM-rescored) numbers:

**1. Train** — writes checkpoints, per-epoch metrics, and the per-sentence
prediction JSON (via the `LogSentencePredictions` callback):

```bash
python -m brain2qwerty_v1.main train
```

**2. Evaluate the checkpoint** on the test split — reloads `best.ckpt` and
saves the test prediction JSON for that exact checkpoint:

```bash
python -m brain2qwerty_v1.main eval --ckpt $BRAIN2QWERTY_RESULTS/best.ckpt
```

**3. Extract a CSV** — turn the callback JSON into a clean, analysis-ready CSV
(one row per sentence, with reconstructed text and CER/WER per subject):

```bash
python -m brain2qwerty_v1.scripts.extract_predictions \
    --input $BRAIN2QWERTY_RESULTS/callbacks --split test --output predictions.csv
```

**4. Rescore with the N-gram LM** — apply the language model to that CSV to
obtain the final LM numbers (see [N-gram decoding](#n-gram-decoding) below).

## N-gram decoding

The keystroke predictions can be modified with a **character-level 9-gram** language model via beam search, as in the paper. This is a post-processing step on the exported predictions; it is not part of training.

**Requirements.** A character-level **9-gram** language model in ARPA (or KenLM binary) format, and the `lm` extra:

```bash
pip install "brain2qwerty[lm]"   # installs kenlm
```

**Where to get the ARPA.** The language model is trained on a freely available public
text corpus (no proprietary data needed). Tokenise the corpus into space-separated
characters (using `&` for spaces) and train a 9-gram with
[KenLM](https://github.com/kpu/kenlm):

```bash
kenlm/build/bin/lmplz -o 9 --text corpus.chars.txt --arpa corpus_9gram.arpa
# optional: binarise for faster loading
kenlm/build/bin/build_binary corpus_9gram.arpa corpus_9gram.bin
```

**Run.** Export predictions first (above), then rescore with the paper's decoding settings — order **9**, beam size **30**, LM weight **5**, max labels per timestep **5**:

```bash
python -m brain2qwerty_v1.scripts.ngram_decoding \
    --input predictions.csv --lm corpus_9gram.arpa --output predictions_with_lm.csv \
    --beam-size 30 --lm-weight 5 --max-labels 5
```

## Reference results

Character error rate (CER) averaged across the 19 MEG participants.

<div align="center">

| Model | CER |
|:-----:|:---:|
| Convolutional Module | 49% |
| **Conv Module + Transformer** | **38%** |

</div>

## Citing

```bibtex
@article{levy2025brain2qwerty,
  title={Non-invasive decoding of typed sentences from human brain activity},
  author={L{\'e}vy, Jarod and Zhang, Mingfang and Pinet, Svetlana and Rapin, J{\'e}r{\'e}my and Banville, Hubert and d'Ascoli, St{\'e}phane and King, Jean-R{\'e}mi},
  journal={Nature Neuroscience},
  year={2025},
  publisher={Nature Publishing Group}
}
```
