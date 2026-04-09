# chemprop-web

A maintained fork of [Chemprop v1.7.1](https://github.com/chemprop/chemprop/tree/v1.7.1) that keeps the browser-based web interface working with modern Python and library versions.

Upstream Chemprop v2 dropped the web app (`chemprop/web/`). This fork preserves it.

**For full documentation** on the model, training options, data formats, and command-line usage, see the [original Chemprop v1.7.1 README](https://github.com/chemprop/chemprop/blob/v1.7.1/README.md).

---

## Installation

```bash
git clone https://github.com/everthoman/chemprop-web.git
cd chemprop-web
pip install -e .
```

Requires Python 3.7–3.11.

## Running the web app

```bash
chemprop_web
```

Then open `http://localhost:5001` in your browser. Use `--host` and `--port` to change the address.

---

## Web interface

The web app supports the full training and prediction workflow through a browser:

### Data

Upload CSV files with a header row. The first column is assumed to be SMILES unless you specify otherwise on the command line. All other columns are treated as targets. Use `--ignore_columns` for identifiers.

### Train

Select a dataset, choose regression or classification, set epochs and ensemble size, and click **Train**. Optionally upload a hyperparameter config JSON (from the Hyperopt page) to train with optimized settings. A scatter plot (regression) or ROC curve (classification) is shown after training.

### Hyperopt

Runs Bayesian hyperparameter optimization (TPE) to find the best model settings for your dataset:

1. Select a dataset and dataset type
2. Set epochs per trial and number of trials (default: 20)
3. Choose which parameters to search:
   - **Basic** — depth, FFN layers, dropout, hidden size *(recommended always)*
   - **Learning rate** — max LR, LR warmup and schedule *(add if Basic alone is insufficient)*
4. Click **Start** — the first half of trials explore randomly, the second half are guided by results so far
5. When complete, download the **config JSON** and upload it on the Train page to train a final model with your chosen epochs and ensemble size

### Predict

Select a trained checkpoint, enter SMILES (typed, drawn, or uploaded as CSV), and click **Predict**. Results can be downloaded as CSV.

---

## Changes relative to Chemprop v1.7.1

### Compatibility

- `torch.load()` calls updated for PyTorch 2.x (`weights_only=False`)
- `argparse.Namespace` registered as a torch safe global for checkpoint loading
- `python_requires` relaxed to `>=3.7,<3.12`
- scikit-learn 1.4+ RMSE fix (`root_mean_squared_error` replaces removed `squared=False`)

### Bug fixes

- Progress bar now correctly tracks all ensemble members (`epochs × ensemble_size`)
- Progress bar `width` CSS property was a quoted string and never updated visually — fixed
- Checkpoint download broken by Werkzeug 2.1 API change — fixed
- Zip checkpoint upload now validates paths against zip slip attacks
- All SQL queries converted to parameterized queries (SQL injection prevention)
- `time.sleep(0)` busy-wait in progress loop replaced with `time.sleep(0.5)`

### Web app additions

- **Hyperparameter optimization page** — Bayesian search (TPE) via hyperopt, with trial progress bar and downloadable config JSON
- **Post-training plots** — scatter plot (regression) and ROC curve (classification) after training
- **Config JSON upload on Train page** — apply hyperopt results at training time
- GPU 0 selected by default when CUDA is available
- Progress bar color transitions red → green during training
- Delete buttons show a confirmation dialog with the item name
- Predictions page shows total count and download link when results are truncated
- Footer version updated to v1.7.1

### Warning suppression

- `PandasTools` import made lazy (eliminates repeated startup warning)
- `hyperopt` and `hyperparameter_optimization` imports wrapped to suppress `pkg_resources is deprecated`
- Numpy deprecation filter broadened in `run_training.py`
