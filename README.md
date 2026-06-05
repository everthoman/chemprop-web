# chemprop-web v1.8.8

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

The web app is password-protected (see [User accounts](#user-accounts) below). Before the first launch, create an account:

```bash
chemprop_web --set_password alice   # prompts for a password, then exits
```

---

## Web interface

The web app supports the full training and prediction workflow through a browser:

### User accounts

The app requires a login. Each user's datasets and checkpoints are private to that account, and the logged-in identity is held in a signed session cookie rather than the client-editable cookie used previously — a user can no longer reach another user's data by editing a cookie.

Credentials are stored **without a database**, in a JSON file (`users_auth.json` in the root folder) mapping each username to a salted password hash (Werkzeug/PBKDF2). Sessions are signed with a key read from the `CHEMPROP_SECRET_KEY` environment variable, or auto-generated and persisted to `.flask_secret_key` on first run so logins survive restarts.

- **Create the first account** from the command line: `chemprop_web --set_password <username>` prompts for a password (and confirmation), writes the hashed credential, and exits without starting the server. Re-run it to change an existing user's password.
- **Add more users** from the running app via the **Create User** page (username + password). This is an **admin-only** action — the page and its navbar link are restricted to admin users. Non-admins are redirected to the home page.
- **Log out** with the link in the top-right navbar, which shows the signed-in username.

**Admin rights.** By default the single user `evehom` is the admin. The admin list is configurable (case-insensitive) via the `CHEMPROP_ADMIN_USERS` environment variable — a comma-separated list of usernames, e.g. `CHEMPROP_ADMIN_USERS=alice,bob`. The `--set_password` CLI is always available to anyone with server shell access regardless of the admin list.

Demo mode (`--demo`) exposes no accounts and is not password-protected.

### Data

Upload CSV files with a header row. The first column must be SMILES; all other columns are treated as targets. If your file contains a non-numeric identifier column (e.g. `chembl_id`), enter its name in the optional **Identifier column** field on upload — it will be excluded from target validation. The same field on the Train page excludes the column from model targets and passes it through to the downloaded train/test predictions CSV. Datasets can be downloaded from the Data page using their display name (e.g. `lipophilicity.csv`). Dataset names can be renamed inline by clicking **Rename** next to any dataset. All datasets for the current user can be removed at once with the **Delete All** button. Datasets are uploaded only from this page (and model checkpoints only from the Checkpoints page); the Train, Hyperopt, and Predict pages just select from what is already uploaded.

### Train

Select a dataset, optionally specify an **identifier column** (a column in your CSV containing compound names or IDs — it will be excluded from targets and passed through to the download CSV), choose regression or classification, set epochs and ensemble size, and click **Train**. Optionally upload a hyperparameter config JSON (from the Hyperopt page) to train with optimized settings.

Optionally enable **early stopping**: tick *Enable early stopping* and set a **Window** (a number of consecutive epochs) and a **Stability band**. Each ensemble member stops once its validation metric stays within the band across the whole window — i.e. the curve has flattened — and keeps training while it still swings by more than the band. The best-validation checkpoint is always kept regardless of where it stops. Disabled by default.

Optionally select **Additional molecule-level features** to augment the graph-based MPNN with precomputed molecular descriptors concatenated to the learned embedding before the output layer:
- **None** (default) — MPNN graph features only.
- **RDKit 2D normalized** (recommended) — 200 CDF-normalised RDKit physicochemical descriptors. No further feature scaling is applied. Requires `descriptastorus` (`pip install descriptastorus`).
- **RDKit 2D** — same 200 descriptors, unnormalised; feature scaling is applied automatically.
- **Morgan binary** — 2048-bit binary Morgan fingerprints (radius 2); feature scaling applied.
- **Morgan count** — 2048-bit count-based Morgan fingerprints (radius 2); feature scaling applied.

Features are saved with the checkpoint and applied automatically during prediction — no manual setup required at predict time. Note that atom contribution maps (hover tooltips) are not available for models trained with molecule-level features, since these descriptors have no per-atom interpretation; a plain structure is shown instead.

Choose a **split type**:
- **Random** (default) — compounds are assigned to train/test randomly (80/10/10).
- **Scaffold** — compounds are split by Bemis-Murcko scaffold so no scaffold present in training appears in the test set. This gives a more honest estimate of generalisation to novel chemical space.

A **Cancel** button is shown during training to stop the run early. A progress bar with an estimated time to completion is shown during training. A live validation convergence chart appears below the progress bar as epochs complete, showing the per-epoch validation metric for each ensemble member. After training:

- **Validation convergence chart** — per-epoch validation metric (e.g. RMSE or AUC) plotted for each ensemble model, shown at the top of the results panel.
- **Regression** — scatter plot (predicted vs experimental) with a per-task statistics table showing R² (train) / Q² (test), RMSE, and MAE for both splits.
- **Classification** — ROC curve with a per-task statistics table (class balance, AUC, Accuracy, Precision, Recall, Specificity, F1, MCC) and a colour-coded confusion matrix (TN/FP/FN/TP), all computed on the test set at a 0.5 threshold.
- **Download train/test predictions** — a CSV containing SMILES, split membership (train/test), experimental values, and predicted values for all compounds. Regression columns: `smiles, split, <task>, pred_<task>`; classification columns: `smiles, split, <task>, pred_prob_<task>`. When ensemble size is greater than 1, a `std_<task>` column is also included showing the ensemble standard deviation for each prediction.

Optionally enable **conformal prediction** with a target error rate **α** (default 0.15 ⇒ 85% coverage). This saves the model's held-out validation split as a calibration set alongside the checkpoint, so predictions made later can carry calibrated [conformal](https://arxiv.org/abs/2107.07511) intervals (regression) or label sets (classification). It works with any ensemble size, including a single model. The training results panel reports, on the independent test split, the **empirical coverage** and mean interval width (regression), or **per-class coverage** for actives and inactives (classification, using class-conditional **Mondrian** calibration so the guarantee holds independently for each class on imbalanced data). Conformal is skipped with a warning if the validation split is too small to calibrate at the chosen α.

Set a **Random seed** (default 666) to control both the train/val/test split and the initial model weights, so a run is fully reproducible; change it to get a different split and a different ensemble.

Defaults: 50 epochs, ensemble size 3, random seed 666.

When training finishes the page shows **Training complete!** right away with the convergence chart and a *Generating plots…* spinner; the scatter/ROC plots and conformal report are computed in the background and appear when ready, so the checkpoint is usable on the Predict page immediately (before the plots finish). The form keeps your last-applied settings (dataset, split type, epochs, ensemble size, early-stopping and conformal options, checkpoint name, etc.) until you change them, so you can see what a finished run was trained on. Results persist when switching tabs; if you navigate away during training and return while it is still running, the progress bar resumes.

### Hyperopt

A **Cancel** button is shown during hyperopt to stop the run early.

Runs Bayesian hyperparameter optimization (TPE) to find the best model settings for your dataset:

1. Select a dataset, optionally specify an **identifier column**, and choose a dataset type
2. Set epochs per trial and number of trials (default: 20)
3. Choose which parameters to search:
   - **Basic** — depth, FFN layers, dropout, hidden size *(recommended always)*
   - **Learning rate** — max LR, LR warmup and schedule *(add if Basic alone is insufficient)*
4. Click **Start** — the first half of trials explore randomly, the second half are guided by results so far
5. When complete, download the **config JSON** (saved as `<dataset_name>_hyperopt.json`) and upload it on the Train page to train a final model with your chosen epochs and ensemble size

The Hyperopt form also keeps your last-applied settings (dataset, trials, search parameters) until you change them.

### Checkpoints

Trained model checkpoints are listed on the Checkpoints page and can be downloaded as a zip file named after the checkpoint (e.g. `lipophilicity_model.zip`). Checkpoint names can be renamed inline by clicking **Rename** next to any checkpoint. All checkpoints for the current user can be removed at once with the **Delete All** button.

Checkpoints trained through the web interface have a **Results** button that opens a modal with the full training results: validation convergence chart, statistics table, and scatter plot (regression) or ROC curve (classification). Hovering over scatter plot points shows the atom contribution heatmap, as on the Train page. A **Download Predictions CSV** button in the modal footer downloads the train/test predictions for that checkpoint. Results are stored permanently alongside the checkpoint file and survive server restarts.

### Predict

Select a trained checkpoint, enter SMILES (typed, drawn, or uploaded as CSV), and click **Predict**. A **Cancel** button is shown during prediction to stop the run early. Results can be downloaded as CSV.

SMILES entered as free text can optionally include a compound identifier separated by a comma, tab, or space (e.g. `CC(=O)Oc1ccccc1C(=O)O, aspirin`). Identifiers are shown alongside predictions in the UI and included as an `id` column in the downloaded CSV. Predicted values are rounded to 3 decimal places in both the UI and the CSV.

Each prediction result includes an **atom contribution map**: a 2D structure overlaid with a Gaussian heatmap showing which atoms increase (green) or decrease (red/pink) the predicted value, computed via gradient × activation (GradCAM-style) and averaged across ensemble members.

When predicting with a checkpoint that has ensemble size greater than 1, each predicted value is shown with a **± std** uncertainty estimate (ensemble standard deviation) in the UI. The downloaded CSV includes a `std_<task>` column alongside each task's prediction.

Optionally enable **conformal prediction** with a target error rate **α** (default 0.15 ⇒ 85% coverage) to attach calibrated, distribution-free uncertainty to each prediction. For regression, each value is shown with a `(1−α)` **prediction interval** and the CSV gains `pi_low_<task>` / `pi_high_<task>` columns; over many molecules the true value is expected to fall inside the interval about `(1−α)` of the time. For classification, conformal uses **class-conditional (Mondrian)** calibration — each class is calibrated separately so the `(1−α)` coverage guarantee holds independently for actives and inactives, which matters on imbalanced QSAR datasets where a marginal guarantee can be met while under-covering the rare actives. Each prediction is labelled **Active** (confidently active), **Inactive** (confidently inactive), **Uncertain** (both plausible), or **Indeterminate** (neither plausible — an atypical molecule), and the CSV gains `conformal_<task>`, `active_in_set_<task>`, and `inactive_in_set_<task>` columns. Conformal takes precedence over the ensemble ± std when both are available. It requires a calibration set saved at training time, so checkpoints trained with conformal disabled (or uploaded without one) fall back to standard predictions with a notice.

Each prediction also carries an **applicability domain** flag indicating whether the molecule resembles the model's training set. For every query molecule the nearest training-set neighbour is found by Tanimoto similarity of 2048-bit Morgan fingerprints (ECFP4); the molecule is labelled **Within domain** or **Outside domain** depending on whether that similarity meets a data-driven cutoff derived from the training set's own nearest-neighbour similarity distribution (5th percentile). A tightly clustered training set produces a stricter cutoff than a chemically diverse one. The UI shows the label plus the nearest-neighbour similarity, and the downloaded CSV adds `ad_similarity` and `ad_in_domain` columns. Predictions flagged *Outside domain* are extrapolations and should be treated with extra caution, independent of the ensemble uncertainty. The flag requires the saved train/test predictions CSV (produced by training in this app) to recover the training SMILES, so checkpoints uploaded without it show no applicability domain.

---

## Changes in v1.8.8

- **Class-conditional (Mondrian) conformal for classification** — classification conformal now calibrates each class separately (computed in the web layer, since chemprop ships only a marginal multilabel calibrator), so the `(1−α)` coverage guarantee holds independently for actives and inactives. This matters on imbalanced QSAR sets, where a single marginal threshold can satisfy the overall guarantee while under-covering the rare actives. Predictions are labelled Active / Inactive / Uncertain / Indeterminate; the Train results report per-class coverage, and the predictions CSV gains `conformal_<task>`, `active_in_set_<task>`, `inactive_in_set_<task>`. Regression conformal is unchanged.
- **"Results first" training** — the Train page now returns the moment training finishes: it shows *Training complete!* with the validation-convergence chart and a *Generating plots…* spinner, while the scatter/ROC plots and conformal report are computed in a background thread and appear when ready. The trained model is usable on the Predict page immediately, before the plots finish. (Fixes along the way: the background prediction no longer crashes on larger datasets — DataLoader workers are disabled in that thread — and the results page no longer auto-starts a second training when it refreshes.)
- **Reworked early stopping** — early stopping is now an explicit **Enable early stopping** checkbox exposing a **Window** (epochs) and a **Stability band**. Training a model stops only once its validation metric stays within the band across the whole window (i.e. the curve has flattened); while it still swings by more than the band, training continues. Each ensemble member decides independently, and the strict-best checkpoint is always kept.
- **Forms remember their settings** — the Train and Hyperopt pages repopulate every field (dataset, identifier column, dataset type, split type, features, epochs, ensemble size, early-stopping and conformal options, checkpoint name, GPU, and the hyperopt trials/search selection) with your last-applied values until you change them, so you can see what a finished run was trained on.
- **Cleaner workflow pages** — data upload lives only on the Data tab and checkpoint upload only on the Checkpoints tab (removed from Train/Hyperopt/Predict); the redundant in-page headings on Train/Hyperopt/Predict were dropped; and the regression predicted-vs-experimental scatter now snaps its axes to whole-unit ticks.
- **Configurable random seed** on the Train page (default 666) controlling both the train/val/test split and the initial model weights, for reproducible runs.
- **Hyperopt config download** is now named `<dataset_name>_hyperopt.json` after the optimized dataset.

## Changes in v1.8.7

- **Conformal prediction on Train and Predict** — an optional, distribution-free uncertainty mode selectable on both pages with a target error rate **α** (default 0.15 ⇒ 85% coverage). Enabling it at training time saves the model's held-out validation split as a calibration set beside the checkpoint; the Train results then report empirical coverage and mean interval width (regression) or per-class coverage (classification) on the independent test set. Classification uses **class-conditional (Mondrian)** conformal so coverage holds independently for actives and inactives on imbalanced data. At predict time, regression predictions gain a `(1−α)` interval (`pi_low_<task>`/`pi_high_<task>` in the CSV) and classification predictions gain an Active/Inactive/Uncertain/Indeterminate conformal label (`conformal_<task>`/`active_in_set_<task>`/`inactive_in_set_<task>`). Works with any ensemble size, including a single model, and takes precedence over the ensemble ± std; checkpoints without a saved calibration set fall back to standard predictions.
- **Password-protected user accounts (no database)** — the web app now requires a login. Per-user credentials are stored as salted password hashes in a JSON file (`users_auth.json`), not in a database; sessions are signed with a persisted secret key (or `CHEMPROP_SECRET_KEY`). The logged-in identity now comes from the signed session instead of the previously client-editable `currentUser` cookie, so users can no longer reach another account's data by editing a cookie. Bootstrap the first account with `chemprop_web --set_password <username>`; creating further users from the **Create User** page is restricted to admins (the user `evehom` by default, overridable via `CHEMPROP_ADMIN_USERS`). Demo mode remains open and unauthenticated.
- **Applicability domain for predictions** — each prediction on the Predict page is now flagged **Within domain** or **Outside domain** based on its Tanimoto similarity (Morgan/ECFP4 fingerprints) to the nearest molecule in the model's training set. The in-/out-of-domain cutoff is derived per-model from the training set's own nearest-neighbour similarity distribution rather than a fixed constant. The downloaded predictions CSV gains `ad_similarity` and `ad_in_domain` columns. Applicability is computed best-effort and only when the checkpoint's saved train/test predictions CSV is available to supply the training SMILES.

## Changes in v1.8.6

- **Target column safety catch** — the Train and Hyperopt pages now scan candidate target columns for non-numeric values before launching a run. If an undeclared identifier column (or any other non-numeric column) is detected, an inline error names the offending column and prompts the user to enter it in the **Identifier column** field or remove it from the CSV, instead of returning a generic 500 error.
- **Identifier column on Hyperopt** — the Hyperopt page now has the same optional **Identifier column** field as Train. When set, the column is excluded from targets via `--ignore_columns` during optimization.

## Changes in v1.8.5

- **Molecule-level features generator** — the Train page now has an **Additional molecule-level features** dropdown. Choosing `rdkit_2d_normalized`, `rdkit_2d`, `morgan`, or `morgan_count` appends precomputed descriptors or fingerprints to the MPNN embedding before the output layer. The selected generator is saved with the checkpoint and applied automatically during prediction and post-training visualisation. Requires `descriptastorus` for the RDKit 2D options (`pip install descriptastorus`).
- **Atom attribution for features models** — models trained with molecule-level features correctly fall back to a plain structure in hover tooltips (no red/green atom colouring), since molecule-level descriptors have no per-atom interpretation. Models trained without features continue to show the GradCAM atom contribution heatmap as before.

## Changes in v1.8.4

- **Ensemble uncertainty** — predictions from multi-model checkpoints now show a **± std** ensemble standard deviation alongside each predicted value in the UI. All downloaded CSVs include a `std_<task>` column: the predict page CSV, the train/test predictions from the Train page, and the Download Predictions CSV from the Checkpoints modal.
- **Download Predictions CSV from Checkpoints** — the Results modal on the Checkpoints page now includes a **Download Predictions CSV** button. The file is saved permanently alongside the checkpoint at training time and remains available after subsequent training runs.
- **Scaffold split** — the Train page now offers a **Scaffold** split type alongside the default **Random** split. Scaffold splitting groups compounds by Bemis-Murcko scaffold and ensures no scaffold seen during training appears in the test set, giving a more honest estimate of generalisation to novel chemical space.
- **Persistent train/test download link** — the Download train/test predictions button on the Train page now links to the permanent per-checkpoint file rather than a temporary file that is overwritten by the next training run.

## Changes in v1.8.3

- **Results modal fix** — the Results button on the Checkpoints page now opens the modal correctly. A quoting bug in the HTML (`tojson` output inside a double-quoted `onclick` attribute) caused clicks to be silently dropped; the modal also now renders outside the page container so Bootstrap z-index behaves correctly.
- **Training completion race condition fixed** — the progress bar and live convergence chart no longer disappear when the browser is minimised and restored during training. The server-side training flag is now cleared only after results are fully prepared, so the page reloads at the right moment.
- **Atom contribution maps in Checkpoints modal** — hovering over scatter plot points in the Results modal shows the GradCAM atom contribution heatmap, identical to the behaviour on the Train page.

## Changes in v1.8.2

- **Early stopping** — optional patience parameter on the Train page; training for each ensemble member stops if the validation metric does not improve for N consecutive epochs. Disabled by default.
- **Consistent decimal formatting** — R², Q², RMSE, and MAE always display 3 decimal places in the statistics table.
- **Robust tab switching** — navigating away from the Train page during training and returning now reliably shows progress while training runs and automatically reloads with results when it finishes, regardless of when you switched tabs.
- **Persistent training results on Checkpoints page** — a Results button appears next to each checkpoint trained through the web interface, opening a modal with the full validation convergence chart, statistics table, and scatter/ROC plot. Results persist across server restarts.

## Changes in v1.8.1

### Predict page
- **Compound identifiers** — free-text SMILES input now accepts an optional identifier (comma, tab, or space separated). Identifiers are displayed in the results and written as an `id` column in the downloaded CSV.
- **Rounded predictions** — predicted values are rounded to 3 decimal places in both the UI and the downloaded CSV.
- **Cancel button** — a Cancel button is shown during prediction to stop the run early.

### Hyperopt page
- **Cancel button** — a Cancel button is shown during hyperopt to stop the run early.

### Data page
- **Rename datasets** — datasets can be renamed inline without leaving the page.
- **Delete All** — removes all datasets for the current user in one click.

### Checkpoints page
- **Rename checkpoints** — checkpoints can be renamed inline without leaving the page.
- **Delete All** — removes all checkpoints for the current user in one click.

### Train page
- **Validation convergence chart** — live line chart of the per-epoch validation metric (one line per ensemble model) appears during training and persists in the results panel; restored from session storage on navigation.
- **ETA during training** — an estimated time to completion is shown below the progress bar, updating every 500 ms based on observed epoch rate.
- **Regression statistics table** — after training, a per-task table shows R² (train) / Q² (test), RMSE, and MAE for both the train and test splits.
- **Classification statistics table** — after training, a per-task table shows class balance, AUC, Accuracy, Precision, Recall, Specificity, F1, and MCC (all at 0.5 threshold) alongside a colour-coded confusion matrix.
- **Identifier column** — optional column name for compound IDs; excluded from targets during training and written as an `id` column in the download CSV.
- **Download train/test predictions** — CSV with SMILES, split membership, experimental values, and predicted values for all compounds.
- **Cancel button** — a Cancel button is shown during training to stop the run early.
- Default epochs changed to 50, default ensemble size to 3.

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
