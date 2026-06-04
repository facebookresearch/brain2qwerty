<!--
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
-->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2025-06-XX

### Added

- Brain2Qwerty decoding pipeline (Conv + Transformer + N-gram LM)
- Training and evaluation code with PyTorch Lightning
- Linear baseline (Ridge classifier with cross-validation)
- Post-training scripts for prediction extraction and N-gram beam search
- MEG and EEG data preprocessing with neuralset
- Grid search launcher for SLURM and local execution