# chemprop-web v1.8.1

A maintained fork of [Chemprop v1.7.1](https://github.com/chemprop/chemprop/tree/v1.7.1) that keeps the browser-based web interface working with modern Python and library versions.

Upstream Chemprop v2 dropped the web app (`chemprop/web/`). This fork preserves and extends it.

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

Select a dataset, optionally specify an **identifier column** (a column in your CSV containing compound names or IDs — it will be excluded from targets and passed through to the download CSV), choose regression or classification, set epochs and ensemble size, and click **Train**. Optionally upload a hyperparameter config JSON (from the Hyperopt page) to train with optimized settings.

A progress bar with an estimated time to completion is shown during training. After training:

- **Regression** — scatter plot (predicted vs experimental) with a per-task statistics table showing R² (train) / Q² (test), RMSE, and MAE for both splits.
- **Classification** — ROC curve with a per-task statistics table (class balance, AUC, Accuracy, Precision, Recall, Specificity, F1, MCC) and a colour-coded confusion matrix (TN/FP/FN/TP), all computed on the test set at a 0.5 threshold.
- **Download train/test predictions** — a CSV containing SMILES, split membership (train/test), experimental values, and predicted values for all compounds. Regression columns: `smiles, split, <task>, pred_<task>`; classification columns: `smiles, split, <task>, pred_prob_<task>`.

Results persist when switching tabs and are restored from session storage when you navigate back.

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

SMILES entered as free text can optionally include a compound identifier separated by a comma, tab, or space (e.g. `CC(=O)Oc1ccccc1C(=O)O, aspirin`). Identifiers are shown alongside predictions in the UI and included as an `id` column in the downloaded CSV. Predicted values are rounded to 3 decimal places in both the UI and the CSV.

Each prediction result includes an **atom contribution map**: a 2D structure overlaid with a Gaussian heatmap showing which atoms increase (green) or decrease (red/pink) the predicted value, computed via gradient × activation (GradCAM-style) and averaged across ensemble members.

---

## Changes in v1.8.1

### Predict page
- **Compound identifiers** — free-text SMILES input now accepts an optional identifier (comma, tab, or space separated). Identifiers are displayed in the results and written as an `id` column in the downloaded CSV.
- **Rounded predictions** — predicted values are rounded to 3 decimal places in both the UI and the downloaded CSV.

### Train page
- **ETA during training** — an estimated time to completion is shown below the progress bar, updating every 500 ms based on observed epoch rate.
- **Regression statistics table** — after training, a per-task table shows R² (train) / Q² (test), RMSE, and MAE for both the train and test splits.
- **Classification statistics table** — after training, a per-task table shows class balance, AUC, Accuracy, Precision, Recall, Specificity, F1, and MCC (all at 0.5 threshold) alongside a colour-coded confusion matrix.
- **Identifier column** — optional column name for compound IDs; excluded from targets during training and written as an `id` column in the download CSV.
- **Download train/test predictions** — CSV with SMILES, split membership, experimental values, and predicted values for all compounds.

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
- **Post-training plots** — scatter plot (regression) and ROC curve (classification) after training; scatter plot persists when switching tabs
- **Hover attribution on scatter plot** — hovering over any train/test point shows its 2D structure with a GradCAM atom contribution map in a floating tooltip
- **Atom attribution maps on Predict page** — per-prediction heatmap showing positive/negative atom contributions (green/red), averaged across ensemble
- **Config JSON upload on Train page** — apply hyperopt results at training time
- GPU 0 selected by default when CUDA is available
- Progress bar color transitions red → green during training
- Delete buttons show a confirmation dialog with the item name
- Predictions page shows total count and download link when results are truncated

### Warning suppression

- `PandasTools` import made lazy (eliminates repeated startup warning)
- `hyperopt` and `hyperparameter_optimization` imports wrapped to suppress `pkg_resources is deprecated`
- Numpy deprecation filter broadened in `run_training.py`
