# chemprop-web

This is a lightly modified fork of [chemprop v1.7.1](https://github.com/chemprop/chemprop/tree/v1.7.1), maintained because the upstream project dropped the web interface in v2.

The original README is preserved below as `README.md`. See the [original repo](https://github.com/chemprop/chemprop) for full documentation.

## Why this fork?

Chemprop v2 is a major rewrite that no longer includes the browser-based web app (`chemprop/web/`). This fork keeps the web app working with modern Python and library versions.

## Changes relative to v1.7.1

### Compatibility fixes

- **`chemprop/utils.py`** — Added `weights_only=False` to all `torch.load()` calls to silence the breaking change introduced in PyTorch 2.x, where the default changed to `weights_only=True`. Affects `load_checkpoint`, `load_frzn_model`, `load_scalers`, and `load_args`.

- **`chemprop/web/run.py`** — Registered `argparse.Namespace` as a safe global via `torch.serialization.add_safe_globals` so that model checkpoints can be loaded without disabling weights-only mode entirely.

- **`setup.py`** — Relaxed `python_requires` from `>=3.7,<3.9` to `>=3.7,<3.12` to allow installation on Python 3.9–3.11.

### Warning suppression

- **`chemprop/features/utils.py`** — Made `from rdkit.Chem import PandasTools` a lazy import (only loaded when `.sdf` files are actually used), eliminating the repeated `Failed to patch pandas - PandasTools will have limited functionality` warning on startup.

- **`chemprop/hyperopt_utils.py`** and **`chemprop/hyperparameter_optimization.py`** — Wrapped `hyperopt` imports in `warnings.catch_warnings()` to suppress the `pkg_resources is deprecated` warning emitted by `hyperopt/atpe.py`.

- **`chemprop/train/run_training.py`** — Broadened the numpy deprecation warning filter from `VisibleDeprecationWarning` to `DeprecationWarning`.

### Web app improvements

- **`chemprop/web/app/templates/train.html`** and **`predict.html`** — GPU 0 is now selected by default in the GPU dropdown when a CUDA device is available, instead of defaulting to CPU (None).
