<!--
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
-->

# Brain2Qwerty

This is the official implementation of [**Non-invasive decoding of typed sentences from human brain activity**](https://doi.org/10.1038/s41593-025-XXXXX) (*Nature Neuroscience*, 2025).

Brain2Qwerty is a three-stage deep neural network that decodes typed sentences from non-invasive brain recordings (MEG and EEG). A convolutional module encodes keystroke-aligned M/EEG windows, a transformer refines predictions at the sentence level, and a character-level N-gram language model corrects the output. With MEG, Brain2Qwerty achieves a character error rate of 29% on average across 19 participants, and as low as 18% for the best participant.

<p align="center">
  <img src="brain2qwerty/resources/approach.png" alt="Brain2Qwerty architecture: convolutional module encodes 500ms M/EEG windows around each keystroke, a transformer refines predictions at the sentence level, and a pretrained N-gram language model corrects the output." width="100%">
</p>

## This repository contains

- The Brain2Qwerty decoding pipeline (Conv + Transformer + N-gram LM) with training and evaluation using PyTorch Lightning
- A per-subject [linear baseline](brain2qwerty/linear_model.py) (Ridge classifier at peak motor-activity latency)
- Post-training scripts for [prediction extraction](brain2qwerty/scripts/extract_predictions.py) and [N-gram beam search decoding](brain2qwerty/scripts/ngram_decoding.py)

## Installation

**Requirements:** Python 3.10+, CUDA-capable GPU.

```bash
git clone https://github.com/facebookresearch/brain2qwerty.git
cd brain2qwerty

# Install bundled dependencies
pip install ./neuralset
pip install "./neuraltrain[lightning]"

# Install brain2qwerty
pip install -e .

# For N-gram language model decoding (optional)
pip install -e ".[lm]"
```

## Data

Due to legal constraints, data will be shared with academic researchers upon request. Please contact svetlana.pinet@univ-lille.fr

### Expected data layout

All data paths are configurable via environment variables:

```bash
export BRAINAI_ROOT="$HOME/brainai"
export BRAINAI_DATA_ROOT="$BRAINAI_ROOT/data"
export BRAINAI_CACHE="$BRAINAI_ROOT/cache"
export BRAINAI_RESULTS="$BRAINAI_ROOT/results"
export BRAINAI_STUDIES_PATH="$BRAINAI_DATA_ROOT/studies"
```

```
$BRAINAI_ROOT/
├── data/
│   ├── studies/               Raw MEG/EEG recordings (neuralset format)
│   │   └── Pinet2024Meg/
│   └── lm_arpa_files/         N-gram language model (.arpa)
│       └── news_9gram.arpa
├── cache/                     Auto-generated preprocessed features
└── results/                   Training outputs and checkpoints
```

### Pre-compute feature cache

Before training, cache the preprocessed M/EEG features (run once per modality):

```bash
python -m brain2qwerty.compute_cache --modality meg
python -m brain2qwerty.compute_cache --modality eeg
```

## Quickstart

### Debug run

Verify the pipeline works end-to-end on a single subject with 2 epochs:

```bash
python -m brain2qwerty.grids.defaults
```

Training should launch without errors, with losses decreasing and no NaN values.

### Full training

Train on all subjects (100 epochs, submitted via SLURM or run locally):

```bash
python -m brain2qwerty.grids.run_grid
```

The grid launcher sweeps over splitting seeds. To customize, edit `brain2qwerty/grids/run_grid.py`.

### Linear baseline

Train a per-subject Ridge classifier at the peak motor-activity time sample (~40 ms post-keystroke):

```bash
python brain2qwerty/linear_model.py
```

## Result extraction and analysis

### Step 1: Extract predictions

The `LogSentencePredictions` callback saves per-sentence predictions during training. Extract them into a CSV:

```bash
python brain2qwerty/scripts/extract_predictions.py \
    --input $BRAINAI_RESULTS/brain2qwerty/<experiment>/callbacks \
    --output predictions.csv
```

### Step 2: Apply N-gram language model

This step requires a character-level 9-gram ARPA language model. We trained ours on the Spanish Wikipedia corpus using [KenLM](https://github.com/kpu/kenlm). To build your own:

```bash
# Install KenLM (see https://github.com/kpu/kenlm for full instructions)
mkdir -p kenlm/build && cd kenlm/build && cmake .. && make -j4

# Train a 9-gram character-level model from a text corpus
./bin/lmplz -o 9 --text corpus.txt --arpa news_9gram.arpa

# Place the model in the expected location
cp news_9gram.arpa $BRAINAI_DATA_ROOT/lm_arpa_files/
```

Then refine predictions with character-level beam search:

```bash
python brain2qwerty/scripts/ngram_decoding.py \
    --input predictions.csv \
    --lm $BRAINAI_DATA_ROOT/lm_arpa_files/news_9gram.arpa \
    --output predictions_with_lm.csv
```

The resulting CSV contains all per-sentence data needed to reproduce the analyses in the paper.

## Reference results

Character error rate (CER) averaged across participants (n=20 for EEG, n=19 for MEG). Each row adds one module on top of the previous.

<div align="center">

| Model | EEG | MEG |
|:-----:|:---:|:---:|
| Conv | 76% | 50% |
| Conv + Transformer | 68% | 32% |
| **Brain2Qwerty** (Conv + Trans + N-gram) | **65 +/- 0.7%** | **29 +/- 1.7%** |

Best MEG participant: 18% CER. Best EEG participant: 61% CER.

</div>

## Citing

If you use this code in your research, please cite:

```bibtex
@article{levy2025brain2qwerty,
  title={Non-invasive decoding of typed sentences from human brain activity},
  author={L{\'e}vy, Jarod and Zhang, Mingfang and Pinet, Svetlana and Rapin, J{\'e}r{\'e}my and Banville, Hubert and d'Ascoli, St{\'e}phane and King, Jean-R{\'e}mi},
  journal={Nature Neuroscience},
  year={2025},
  publisher={Nature Publishing Group}
}
```

## Dependencies

This repository ships pinned versions of [neuralset](neuralset/) and [neuraltrain](neuraltrain/) for reproducibility. The latest maintained versions of these libraries are available as part of the [NeuroAI](https://github.com/facebookresearch/neuroai) suite.

## Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License

Brain2Qwerty is released under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](LICENSE) license.