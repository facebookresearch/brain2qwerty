# neuraltrain

`neuraltrain` is a lightweight library of useful classes and functions for training deep learning models with neuralset.

## Install

If you have already installed `neuralset` following its installation instructions, you can just do:

```
conda activate <your_neuralset_env>
pip install --config-settings editable_mode=strict -e .
```
*Note*: `editable_mode=strict` is required for `mypy` to pick up `neuralset` typing from within `neuraltrain` when installed as editable (`-e`).

If starting from scratch:

```
conda create -n neuraltrain python=3.10 ipython -y
conda activate neuraltrain
pip install -e .
```

To install additional `dev`, `lightning` and `models` dependencies, you can instead run: 
```
pip install --config-settings editable_mode=strict -e .'[dev,lightning,models]'
```

*Note*: if you get errors when install lightning (caused by fastAPI), try explicitly requesting a version: `pip install lightning==2.5.0.post0`