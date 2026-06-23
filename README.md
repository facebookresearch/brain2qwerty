# Brain2Qwerty

Decoding the sentences a person types from their non-invasive brain activity.
Each participant types short sentences in a MEG scanner; a deep learning model
learns to map the resulting brain signals back to text.

The code release lives in [`brain2qwerty/`](brain2qwerty/) — see its
[README](brain2qwerty/README.md) for installation, data, training and
evaluation of both model versions:

- **V1** — keystroke-level decoding with a convolutional + transformer encoder.
- **V2** — sentence-level end-to-end decoding (CTC + word-level contrastive
  alignment + a LoRA-adapted language model).
