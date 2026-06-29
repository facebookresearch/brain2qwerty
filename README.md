<!--
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
-->

# Brain2Qwerty

<p align="center">
  <img src="resources/comm_hero_clean.gif" alt="A participant types in an MEG scanner; Brain2Qwerty reconstructs the sentence from brain activity." width="100%">
</p>

The project Brain2Qwerty studies whether it is possible to decode the sentences a
person types from their brain activity alone. Each participant of the study types
short sentences in a MEG scanner. This scanner captures the magnetic field elicited
by the brain. A deep learning model learns to map the resulting brain signals back
to text.

A high-level overview of the project is available on the [project website](https://facebookresearch.github.io/brain2qwerty/).

This repository bundles the two pipelines of the Brain2Qwerty line of work, each in
its own self-contained package with models, training and evaluation code:

- **[`brain2qwerty_v1/`](brain2qwerty_v1/)** — [*Non-invasive decoding of typed sentences from human brain activity*](https://www.nature.com/articles/s41593-026-02303-2) (Nature Neuroscience, 2026). A convolutional encoder and a sentence-level transformer decode one keystroke at a time, synchronised to keystroke onsets.

- **[`brain2qwerty_v2/`](brain2qwerty_v2/)** — [*Accurate Decoding of Natural Sentences from Non-Invasive Brain Recordings*](https://ai.meta.com/research/publications/accurate-decoding-of-natural-sentences-from-non-invasive-brain-recordings/) (under review, 2026). An end-to-end model decodes whole sentences from a single continuous recording window, combining a CTC encoder, a word-level contrastive aligner, and a LoRA-adapted language model.

Each package ships its own README with installation, data, training and evaluation
instructions. Both packages rely extensively on the public libraries
[neuralset](https://github.com/facebookresearch/neuroai/tree/main/neuralset-repo) and
[neuraltrain](https://github.com/facebookresearch/neuroai/tree/main/neuraltrain-repo).

If you are working with an AI coding agent, point it to
[AGENT_README.md](AGENT_README.md) for a deep dive into the codebase.

## License

The code is released under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](LICENSE) license. The datasets belong to the [BCBL — Basque Center on Cognition, Brain and Language](https://www.bcbl.eu/).
