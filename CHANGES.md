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

### Bug fixes

- **`chemprop/train/metrics.py`** — Replaced `mean_squared_error(..., squared=False)` with `root_mean_squared_error`, fixing a crash with scikit-learn 1.4+ which removed the `squared` parameter.

- **`chemprop/web/app/views.py`** — Fixed progress bar to track all ensemble members correctly. Previously it counted epochs against a single model's total, completing at ~10% when training an ensemble of 10. Now counts total epoch log entries against `epochs × ensemble_size`.

- **`chemprop/web/app/views.py`** — Replaced `time.sleep(0)` busy-wait with `time.sleep(0.5)` in the progress tracking loop, preventing a CPU thread from being pegged during training.

- **`chemprop/web/app/views.py`** — Fixed checkpoint download: `attachment_filename=` (removed in Werkzeug 2.1) replaced with `download_name=`.

- **`chemprop/web/app/views.py`** — Zip checkpoint uploads now validate extracted paths to prevent zip slip (path traversal) attacks.

- **`chemprop/web/app/db.py`** and **`views.py`** — Replaced all f-string SQL queries with parameterized queries to prevent SQL injection.

- **`chemprop/web/app/templates/train.html`** — Fixed progress bar CSS `width` property which had a quoted string value (`"{{ progress }} %"`) preventing it from updating visually.

- **`chemprop/web/app/views.py`** — Fixed a 500 on the Train page whenever auto-binarize was enabled. The dataset lookup was changed to resolve through the registry, so the name in hand became the row's integer id, and `secure_filename` rejects one (`TypeError: normalize() argument 2 must be str, not int`). The binarized copy is now written through `user_temp_path`, which also stops two people binarizing the same dataset from overwriting each other's file while their runs are reading it.

- **`chemprop/web/app/views.py`** and **`backends.py`** — The Train page's "Judge on" choice, offered for chemprop 2 runs, was read from the form and then never used: early stopping and checkpoint selection always followed the task type's default metric, so a classification run was judged on AUC whatever the dropdown said. The choice now reaches `--tracking-metric`.

- **`chemprop/web/app/views.py`** and **`workers.py`** — Training and hyperopt workers moved to a `spawn` subprocess, fixing the CUDA "Cannot re-initialize CUDA in forked subprocess" error, and a `process.join()` that hung after training finished.

- **`chemprop/web/app/views.py`** — Progress reporting fixed repeatedly: stalling on the second and later runs of a session, a `/receiver` 500 that froze the display outright, and a multi-worker gunicorn setup where the polling request reached a process that knew nothing about the run.

- **`chemprop/web/app/views.py`** — Training results are no longer lost when switching tabs mid-run, and reloading the results page no longer starts the training over. Logging out during training, and hyperopt reporting itself as running while a training job was active, were both fixed.

- **`chemprop/web/app/views.py`** — The applicability domain no longer depends on the order of rows in the prediction input.

- **`chemprop/web/app/views.py`** — Blank column names produced by a spreadsheet's trailing commas (`smiles,logP,id,,,,`) are dropped when deriving task names, instead of being passed to the model as unnamed targets. Hyperopt also excludes a declared identifier column from the targets, which Train had always done.

- **`chemprop/web/app/views.py`** — Uploads are no longer rejected as an expired page, and a 500 on the Hyperopt page and when training on a reused split were fixed.

- **`chemprop/web/app/backends.py`** — chemprop 2 runs are no longer cut short by the early-stopping band. The two backends read the same form fields differently: chemprop 1 stops once the last N scores sit inside a `min_delta` band, while chemprop 2 hands `min_delta` to Lightning as the improvement each epoch must deliver. Passing the form's 0.01 default through demanded a loss improvement of 0.01 every epoch. On BBBP with patience 5 that ended training at 14 epochs and AUC 0.831, against 40 epochs and AUC 0.916 without it. Models trained on the chemprop 2 backend with early stopping before this fix are worth retraining.

- **`chemprop/web/app/backends.py`** — chemprop 2 classification runs are judged on AUC rather than the cross-entropy loss, which flattens while ranking is still improving, so a run stopped with its AUC still climbing and kept the best-loss epoch instead of the best-AUC one.

- **`chemprop/web/app/views.py`** — Bounded the structure cache by size, cleared the mol-graph cache after the training visualization, and moved background plotting off DataLoader worker processes, which had made it slow or hang.

- **`chemprop/web/app/templates/`** — Chart.js 4.x fixes: the regression scatter plot not rendering in a mixed chart, the ROC legend drawing "Random" as a box rather than a dashed line, and metric values always showing three decimal places.

### Web app: training

- **Hyperparameter optimisation** — Added to the web app, with the search space selectable by keyword and the resulting configuration downloadable and feedable back into a training run.

- **Early stopping** — Added as a patience option, then gated behind an Enable checkbox so it can be switched off cleanly, given a minimum-improvement (`min_delta`) threshold, and switched to a plateau-band criterion on the chemprop 1 backend: training stops once the last N validation scores all sit inside the band, rather than on the first non-improving epoch. Default patience moved from 5 to 10, since a noisy validation curve needs a longer window before a run is abandoned.

- **What early stopping is judged on** — For chemprop 2 runs, a choice between the plotted metric and the validation loss, which also decides which epoch's model is kept.

- **Split control** — A scaffold-balanced split option; configurable train/validation/test sizes that reach both backends and are refused if they do not add up; and the ability to reuse an earlier run's split so that two models are comparable, since the backends partition differently even from the same seed. Runs made before splits were recorded are still offered where their partition can be recovered.

- **Random seed** — Exposed on the Train page, default 666, so a run can be repeated exactly.

- **Molecule-level features** — A generator selector (RDKit 2D, normalized or not; Morgan binary or count fingerprints), mapped to the chemprop 2 equivalent where one exists.

- **Identifier columns** — A named identifier column is carried through upload validation, training and prediction without being handed to the model as a target, and target columns are validated up front to catch an undeclared one.

- **Auto-binarize** — Continuous activity measurements can be thresholded into active/inactive labels for a classification run, by median + k x MAD, by percentile, or at a fixed value, reporting the threshold and the resulting class balance. A threshold that puts every compound in one class is refused.

- **Y-scramble** — A control run that permutes the target columns before the split, leaving the structures and the block of target values intact and destroying only the pairing between them. A run on the copy has no structure-activity signal to learn and should score at chance; one that scores well is measuring something else. Available for regression and classification on both backends, and the checkpoint is named `[y-scrambled]` so a control cannot be mistaken for a usable model. On the 4200-row lipophilicity set, identical settings give test R² +0.68 for the real run and −0.00 for the scrambled one.

- **Conformal prediction** — Available on training and prediction, class-conditional (Mondrian) for classification, with the miscoverage rate configurable and defaulting to 0.15 (85% coverage).

- **Cancel** — Train, Hyperopt and Predict runs can be cancelled, and the button survives navigating away and back.

### Web app: results and analysis

- **Plots** — A regression scatter plot and a classification ROC curve after training, and a live validation convergence chart during it. Heavy plotting moved to a background thread so the page shows results as soon as training ends and fills the plots in behind.

- **Statistics** — R2, RMSE and MAE for regression; AUC, accuracy, precision, recall, specificity, F1 and MCC for classification, with a confusion matrix and the class balance of the split.

- **Atom attribution maps** — Per-atom contributions rendered on the structure, for both backends, with a chemprop 2 worker kept alive between requests so the first map is not slow.

- **Applicability domain** — Predictions carry a training-set similarity measure and an `ad_threshold` column, flagging compounds the model was not trained near.

- **Downloadable predictions** — A combined train/test predictions CSV, ensemble standard deviation alongside each prediction, and training results that persist on the Checkpoints page after the session that produced them.

### Web app: the chemprop 2 backend

The app runs on chemprop 1.x and shells out to a chemprop 2 installation in a separate conda environment, so this process never imports chemprop 2.

- **Foundation-model finetuning** — Training from CheMeleon, with the foundation model optional within the backend, and chemprop 2 as the default backend where its environment is present.

- **Coverage** — Hyperparameter optimisation, atom attribution, molecule structure rendering, checkpoint upload, and starting a run from an earlier checkpoint all work on chemprop 2 checkpoints, which chemprop 1.x cannot open.

- **Batch size** — Exposed on the Train page, kept at chemprop 2's own default of 64: raising it was measured to buy no wall-clock time while costing validation loss at a fixed epoch budget.

- **Cancellation** — A killed chemprop 2 run is recognised as cancelled rather than successful. Lightning traps `SIGTERM` and exits 0, so the exit code cannot be used.

### Web app: multiple users and access control

- **Authentication** — File-based per-user passwords with admin rights, replacing an open app.

- **Ownership** — Every route that takes a dataset or checkpoint id checks that it belongs to the requesting user, so an id cannot be guessed into someone else's data.

- **Per-user state** — Job state, last-used form settings, and temporary result files are scoped per user; results used to be written to one shared name, so whoever asked for a download next got whichever run finished last.

- **Sessions** — An expired session returns 401 JSON to AJAX calls instead of a login page the JavaScript could not read.

- **Security review fixes** — An authentication bypass, path traversal, zip slip on checkpoint upload, a cache race, SQL built by f-string, and CSV parsing bugs.

- **Managing datasets and checkpoints** — Rename, delete with a confirmation naming the item, and Delete All.

### Web app: deployment and hardware

- **Host and port** — Configurable via `CHEMPROP_HOST` and `CHEMPROP_PORT`.

- **GPU selection** — Only cards large enough to train on are offered, named with their model and memory, and the list means the same thing on any host: CUDA orders devices fastest-first while `nvidia-smi` orders them by PCI bus, so the index chosen in the dropdown could select a different card than its label named.

- **PyTorch version warning** — The pinned PyTorch predates current GPUs, and the app now says so rather than failing obscurely on one.

- **Reproducing the deployment** — The exact environments the deployment runs on are recorded, with notes on what installing it on a second machine takes. The runtime temp folder is no longer tracked in git.

### Scripts

- **`scripts/`** — CLI batch prediction with conformal intervals, ensemble uncertainty and Tanimoto applicability domain, and a pipeline for building a descriptor-pretraining corpus.

### Web app: interface details

- **`chemprop/web/app/templates/train.html`** and **`predict.html`** — GPU 0 is now selected by default in the GPU dropdown when a CUDA device is available, instead of defaulting to CPU (None).

- **`chemprop/web/app/templates/train.html`** — Progress bar color transitions from red to green as training progresses using HSL color interpolation.

- **`chemprop/web/app/templates/checkpoints.html`** and **`data.html`** — Delete buttons now show a confirmation dialog naming the item before permanently deleting it.

- **`chemprop/web/app/templates/predict.html`** — Truncated predictions notice replaced with a visible Bootstrap alert showing the total prediction count and a direct link to download the full CSV.