"""Defines a number of routes/views for the flask app."""

from functools import wraps
import csv
import io
import math
import os
import re
import sys
import shutil
from contextlib import nullcontext
from tempfile import TemporaryDirectory, NamedTemporaryFile
import threading
import time
from typing import Callable, List, Optional, Tuple
import multiprocessing as mp
import zipfile

from flask import json, jsonify, redirect, render_template, request, send_file, send_from_directory, session, url_for
import numpy as np
from rdkit import Chem
import torch
from werkzeug.utils import secure_filename

from chemprop.web.app import app, auth, db

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from chemprop.args import PredictArgs, TrainArgs, HyperoptArgs
from chemprop.constants import MODEL_FILE_NAME, TRAIN_LOGGER_NAME
from chemprop.data import get_data, get_header, get_smiles, get_task_names, validate_data, split_data, empty_cache
from chemprop.train import make_predictions, run_training
from chemprop.utils import create_logger, load_task_names, load_args
from chemprop.web.app.atom_attribution import (compute_attributions, plain_svg,
                                               render_attribution_svg)
from chemprop.web.app.applicability import ApplicabilityDomain, load_training_smiles
from chemprop.web.app import backends

class Job:
    """One user's running training, hyperopt or prediction.

    This state used to be a single set of module globals, so two jobs running at
    once overwrote each other: the progress bar showed a mixture of both and
    /cancel killed whichever had started most recently rather than the one the
    user was looking at. Keyed by user instead, each job owns its own state.

    ``progress`` stays a shared value because the progress bar runs in a forked
    process; everything else is only touched by the request thread.
    """

    def __init__(self, mode: str, user_key: str):
        self.mode = mode          # 'train', 'hyperopt' or 'predict'
        self.user_key = user_key
        self.progress = mp.Value('d', 0.0)
        self.process = None       # training/hyperopt/prediction subprocess
        self.progress_bar = None
        self.log_path = ''
        self.v2_dir = ''          # output dir of a chemprop 2.x run, for live curves
        self.cancelled = False

    def is_running(self) -> bool:
        return self.process is not None and self.process.is_alive()

    def stop(self) -> None:
        """Terminates the job's processes and marks it cancelled.

        The flag is raised before the processes are signalled: the view waiting on
        the job notices the process dying and reads this flag immediately, so
        setting it afterwards let it read False and report a failure instead.
        """
        self.cancelled = True
        for proc in (self.process, self.progress_bar):
            if proc and proc.is_alive():
                proc.terminate()
                proc.join(timeout=3)
                if proc.is_alive():
                    proc.kill()
        self.process = None
        self.progress_bar = None


JOBS = {}  # user key -> Job
JOBS_LOCK = threading.Lock()


def job_key() -> str:
    return str(current_user_id() or '')


def start_job(mode: str) -> "Job":
    """Registers a new job for the current user, replacing any finished one."""
    key = job_key()
    job = Job(mode, key)
    with JOBS_LOCK:
        JOBS[key] = job
    return job


def current_job():
    """The current user's job, or None."""
    return JOBS.get(job_key())


def end_job(job: "Job") -> None:
    """Removes a job once its results have been prepared."""
    with JOBS_LOCK:
        if JOBS.get(job.user_key) is job:
            del JOBS[job.user_key]
LAST_TRAIN_RESULT = {}  # user_key -> last successful training kwargs, served on GET after tab switch
LAST_TRAIN_SETTINGS = {}     # user_key -> last submitted Train form values, to repopulate the form
LAST_HYPEROPT_SETTINGS = {}  # user_key -> last submitted Hyperopt form values


def _normalize_to_csv(path: str) -> None:
    """Re-write a tab- or whitespace-separated file as a comma-separated CSV in place.

    Detects the delimiter from the first line by whichever of tab/comma occurs more
    often (ties favor tab), then falls back to splitting on any whitespace (handles
    multi-space-separated files). Strips a UTF-8 BOM if present, filters trailing
    blank rows, and writes Unix line endings.
    """
    try:
        with open(path, 'rb') as f:
            has_bom = f.read(3) == b'\xef\xbb\xbf'
        enc = 'utf-8-sig' if has_bom else 'utf-8'

        with open(path, newline='', encoding=enc) as f:
            first_line = f.readline()
    except UnicodeDecodeError:
        with open(path, 'rb') as f:
            has_bom = f.read(3) == b'\xef\xbb\xbf'
        enc = 'utf-8-sig' if has_bom else 'latin-1'

        with open(path, newline='', encoding=enc) as f:
            first_line = f.readline()

    tab_count = first_line.count('\t')
    comma_count = first_line.count(',')

    if tab_count > 0 and tab_count >= comma_count:
        delimiter = '\t'
    elif comma_count > 0 and not has_bom:
        return  # already clean comma-separated CSV
    else:
        delimiter = ',' if comma_count > 0 else None  # BOM-only CSV or whitespace-split

    rows = []
    with open(path, newline='', encoding=enc) as f:
        if delimiter:
            for row in csv.reader(f, delimiter=delimiter):
                if any(cell.strip() for cell in row):  # skip blank rows
                    rows.append(row)
        else:
            for line in f:
                stripped = line.rstrip('\r\n')
                if stripped:
                    rows.append(stripped.split())

    # Pad rows shorter than the header with empty strings (missing values → treated as NaN).
    if rows:
        n_cols = len(rows[0])
        rows = [row + [''] * (n_cols - len(row)) for row in rows]

    with open(path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f, lineterminator='\n').writerows(rows)


def _find_non_numeric_columns(data_path: str, columns: List[str], sample_rows: int = 200) -> List[str]:
    """Scan a CSV's columns and return those containing any non-numeric values.

    Used to detect undeclared identifier columns before chemprop's get_data() tries
    to call float() on them and raises a generic ValueError.
    """
    if not columns:
        return []
    bad = []
    try:
        with open(data_path) as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= sample_rows:
                    break
                for col in columns:
                    if col in bad:
                        continue
                    value = (row.get(col) or '').strip()
                    if value in ('', 'nan'):
                        continue
                    # chemprop accepts inequality and list-encoded targets
                    if value[0] in '<>[':
                        continue
                    try:
                        float(value)
                    except ValueError:
                        bad.append(col)
    except (OSError, csv.Error):
        return []
    return bad


def _binarize_csv(data_path: str, task_names: List[str], method: str,
                  param: float, out_path: str) -> List[dict]:
    """Write a copy of data_path with target columns binarized (active=1 if value >= threshold).

    method: 'mad'        -> threshold = median + param * MAD
            'percentile' -> threshold = param-th percentile
            'fixed'      -> threshold = param

    Returns list of {name, threshold, n_active, n_inactive, n_total} per task.
    """
    with open(data_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    stats = []
    for col in task_names:
        vals = []
        for row in rows:
            v = (row.get(col) or '').strip()
            if v:
                try:
                    vals.append(float(v))
                except ValueError:
                    pass

        if not vals:
            stats.append({'name': col, 'threshold': None, 'n_active': 0, 'n_inactive': 0, 'n_total': 0})
            continue

        arr = np.array(vals)
        if method == 'mad':
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            threshold = med + param * mad
        elif method == 'percentile':
            threshold = float(np.percentile(arr, param))
        else:
            threshold = float(param)

        n_active = n_inactive = 0
        for row in rows:
            v = (row.get(col) or '').strip()
            if v:
                try:
                    b = 1 if float(v) >= threshold else 0
                    row[col] = str(b)
                    if b:
                        n_active += 1
                    else:
                        n_inactive += 1
                except ValueError:
                    pass

        stats.append({'name': col, 'threshold': round(threshold, 4),
                      'n_active': n_active, 'n_inactive': n_inactive,
                      'n_total': n_active + n_inactive})

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return stats


def _parse_val_curves(log_path):
    """Parse verbose.log and return per-model validation scores by epoch."""
    models = []
    current_scores = None
    current_epoch = None
    epoch_seen = set()
    metric_name = None

    try:
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if re.match(r'^(Building|Loading) model \d+', line):
                    current_scores = []
                    models.append(current_scores)
                    epoch_seen = set()
                    current_epoch = None
                    continue
                m = re.match(r'^Epoch (\d+)$', line)
                if m:
                    current_epoch = int(m.group(1))
                    continue
                m = re.match(r'^Validation (\w+) = ([\d.eE+\-]+)$', line)
                if m and current_epoch is not None and current_epoch not in epoch_seen:
                    if current_scores is None:
                        current_scores = []
                        models.append(current_scores)
                    if metric_name is None:
                        metric_name = m.group(1)
                    if m.group(1) == metric_name:
                        current_scores.append(float(m.group(2)))
                        epoch_seen.add(current_epoch)
    except Exception:
        pass

    return {'metric': metric_name or '', 'models': models}


from chemprop.web.app.workers import train_worker as _train_worker
from chemprop.web.app.workers import hyperopt_worker as _hyperopt_worker
from chemprop.web.app.workers import predict_worker as _predict_worker

# Use spawn so CUDA can be initialized fresh in each worker subprocess.
# Progress-bar subprocesses stay on the default (fork) context because they
# never touch CUDA and forking them is safe and cheap.
_spawn = mp.get_context('spawn')


def conformal_calibration_path(ckpt_id) -> str:
    """Path to a checkpoint's saved conformal calibration set (held-out val split).

    Its presence is what makes conformal prediction available for a checkpoint
    at predict time; uploaded checkpoints won't have one.
    """
    return os.path.join(app.config['CHECKPOINT_FOLDER'], f'{ckpt_id}_calibration.csv')


def parse_conformal_form(form) -> Tuple[bool, float]:
    """Read the conformal checkbox + alpha from a submitted form.

    Returns (enabled, alpha) with alpha clamped to the open interval (0, 1) and
    defaulting to 0.15 (i.e. 85% target coverage).
    """
    enabled = form.get('conformalEnabled', 'False') == 'True'
    raw = (form.get('conformalAlpha', '') or '').strip()
    try:
        alpha = float(raw) if raw else 0.15
    except ValueError:
        alpha = 0.15
    if not (0 < alpha < 1):
        alpha = 0.15
    return enabled, alpha


def conformal_calibration_feasible(n_cal: int, alpha: float) -> bool:
    """Whether a calibration set of size ``n_cal`` can yield a finite conformal quantile.

    chemprop computes ``q_level = ceil((n+1)(1-alpha)) / n`` and takes that quantile of
    the calibration scores; it must be <= 1 for the quantile to be well defined.
    """
    return n_cal >= 1 and math.ceil((n_cal + 1) * (1 - alpha)) <= n_cal


def mondrian_conformal_thresholds(cal_probs, cal_labels, alpha):
    """Class-conditional (Mondrian) conformal thresholds for binary classification.

    QSAR datasets are typically imbalanced, where a single marginal conformal
    threshold can satisfy its guarantee on the majority (inactive) class while
    under-covering the rare actives. Mondrian conformal calibrates a separate
    threshold per class so ~(1-alpha) coverage holds independently within actives
    and inactives.

    Score is ``1 - p_trueclass``; per class ``c`` the threshold ``q_c`` is the
    ``ceil((n_c+1)(1-alpha))/n_c`` empirical quantile of that class's calibration
    scores. A query's set then includes class ``c`` iff ``1 - p_c <= q_c``.

    :param cal_probs: per-datapoint list of per-task P(active).
    :param cal_labels: per-datapoint list of per-task true label (0/1, or None).
    :return: per-task dict with q_active/q_inactive (None if no calibration points
             for that class) and the per-class calibration counts.
    """
    if not cal_probs:
        return []
    n_tasks = max((len(r) for r in cal_probs if r is not None), default=0)

    def _qhat(scores):
        n = len(scores)
        if n == 0:
            return None
        q_level = math.ceil((n + 1) * (1 - alpha)) / n
        if q_level >= 1.0:
            return 1.0  # too few points to calibrate -> include the class always
        return float(np.quantile(scores, q_level, method='higher'))

    per_task = []
    for t in range(n_tasks):
        s_active, s_inactive = [], []
        for i in range(len(cal_probs)):
            if cal_probs[i] is None or cal_labels[i] is None:
                continue
            y = cal_labels[i][t] if t < len(cal_labels[i]) else None
            if y is None:
                continue
            try:
                p = float(cal_probs[i][t])
            except (TypeError, ValueError, IndexError):
                continue
            if int(round(float(y))) == 1:
                s_active.append(1.0 - p)
            else:
                s_inactive.append(p)
        per_task.append({
            'q_active': _qhat(s_active), 'q_inactive': _qhat(s_inactive),
            'n_active': len(s_active), 'n_inactive': len(s_inactive),
        })
    return per_task


def mondrian_category(p, thr):
    """Conformal label for one binary task given P(active) and its thresholds.

    Returns 'active'/'inactive' (confident), 'uncertain' (both classes plausible),
    or 'none' (neither plausible at this confidence — an atypical point).
    """
    try:
        p = float(p)
    except (TypeError, ValueError):
        return None
    qa, qi = thr.get('q_active'), thr.get('q_inactive')
    active_in = qa is not None and (1.0 - p) <= qa
    inactive_in = qi is not None and p <= qi
    if active_in and not inactive_in:
        return 'active'
    if inactive_in and not active_in:
        return 'inactive'
    if active_in and inactive_in:
        return 'uncertain'
    return 'none'


FOUNDATION_CKPT_PREFIX = 'ckpt:'


def foundation_checkpoints(user_id):
    """The user's chemprop 2 checkpoints, offered as starting points for a new run.

    chemprop 2 accepts a path to any of its own models for --from-foundation and
    reuses that model's message passing, so a model trained here on a large set can
    seed a run on a small one — often a better starting point than a generic
    foundation model, since it already knows the chemistry in question.
    """
    rows = db.query_db(
        "SELECT id, ckpt_name FROM ckpt WHERE associated_user = ? AND backend = 'v2' "
        "ORDER BY id DESC", (user_id or app.config['DEFAULT_USER_ID'],))
    return [row for row in rows if db.get_models(row['id'])]


def resolve_foundation(value):
    """Maps a submitted foundation choice to ``(cli_value, label)``.

    ``cli_value`` is what --from-foundation receives: a registry name, or the path
    to one of the current user's own models. Raises ValueError with a message to show if
    the choice cannot be honoured.
    """
    if not value:
        return None, None

    if value in app.config['FOUNDATION_MODELS']:
        return value, value

    if not value.startswith(FOUNDATION_CKPT_PREFIX):
        raise ValueError(f'Unknown foundation model "{value}".')

    try:
        ckpt_id = int(value[len(FOUNDATION_CKPT_PREFIX):])
    except ValueError:
        raise ValueError('That starting checkpoint is not valid.')

    # Only the user's own chemprop 2 checkpoints, so a submitted id cannot reach
    # another user's models or an arbitrary path.
    row = owned_ckpt(ckpt_id)
    if row is None or row['backend'] != 'v2':
        raise ValueError('That starting checkpoint is no longer available.')

    models = db.get_models(ckpt_id)
    if not models:
        raise ValueError(f'Checkpoint "{row["ckpt_name"]}" has no model files to start from.')

    # chemprop takes a single message passing block, so an ensemble contributes
    # its first member.
    path = os.path.join(app.config['CHECKPOINT_FOLDER'], f'{models[0]["id"]}.pt')
    if not os.path.exists(path):
        raise ValueError(f'The model file for "{row["ckpt_name"]}" is missing.')

    return path, f'checkpoint: {row["ckpt_name"]}'


def v2_checkpoint_info_script() -> str:
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'v2_checkpoint_info.py')


def validate_uploaded_v2(info, raw_task_names: str):
    """Checks an uploaded chemprop 2 checkpoint and settles its target names.

    The checkpoint records how many targets it predicts and what kind of task they
    are, but not what they are called, so the names come from the upload form. A
    single-target model is named for the user if they did not say.
    """
    if info is None:
        raise ValueError('That file is not a checkpoint this app can read — it is '
                         'neither a chemprop 1 nor a chemprop 2 model.')

    if info.get('multicomponent'):
        raise ValueError('This is a multi-molecule chemprop 2 model, which this app '
                         'does not support.')

    n_tasks = info['n_tasks']
    task_names = [name.strip() for name in raw_task_names.split(',') if name.strip()]

    if not task_names:
        if n_tasks == 1:
            return ['prediction']
        # A foundation model predicts hundreds of descriptors whose names carry no
        # meaning for this app, and nobody should have to type them in to register
        # one; they are only ever used as CSV headers.
        return [f'target_{i + 1}' for i in range(n_tasks)]

    if len(task_names) != n_tasks:
        raise ValueError(f'This model predicts {n_tasks} target(s) but {len(task_names)} '
                         f'name(s) were given.')

    return task_names


def v2_attribution_script() -> str:
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'v2_attribution.py')


def warm_v2_attribution(ckpt_id) -> None:
    """Starts the attribution worker while a v2 model's plots are being opened.

    The worker takes a few seconds to import chemprop 2. Starting it when the page
    that will hover those structures is served means the first hover is as quick
    as the rest.
    """
    if not app.config['CHEMPROP2_AVAILABLE'] or ckpt_backend(ckpt_id) != 'v2':
        return
    backends.attribution_worker(
        app.config['CHEMPROP2_PYTHON'], v2_attribution_script(),
        backends.subprocess_env(None),
        os.path.join(app.config['TEMP_FOLDER'], 'v2_attribution.log')).warm()


def v2_attribution_svgs(ckpt_id, model_paths, smiles_list, gpu=None):
    """Attribution SVGs for a chemprop 2 checkpoint, as ``(svgs, attributed)``.

    The weights come from a helper run under the v2 interpreter (this process
    cannot load a v2 checkpoint); the drawing is shared with the v1 path, so both
    backends colour a structure the same way. Molecules whose weights could not be
    computed fall back to a plain depiction rather than to nothing.
    """
    weights = backends.atom_weights(
        app.config['CHEMPROP2_PYTHON'], v2_attribution_script(), model_paths, smiles_list,
        backends.subprocess_env(gpu),
        stderr_path=os.path.join(app.config['TEMP_FOLDER'], 'v2_attribution.log'))

    svgs, attributed = [], []
    for smiles, w in zip(smiles_list, weights):
        svg = render_attribution_svg(smiles, np.array(w, dtype=float)) if w else None
        attributed.append(svg is not None)
        svgs.append(svg if svg is not None else plain_svg(smiles))
    return svgs, attributed


def _v2_valid_mask(smiles):
    """Which query entries RDKit can parse.

    chemprop 2.x raises on the first unparseable SMILES and produces nothing,
    whereas the v1 backend returns None for that row and the page labels it
    "Invalid SMILES String". Invalid entries are therefore filtered out before the
    CLI sees them, and restored as None afterwards, so one bad paste does not lose
    the whole prediction.
    """
    return [Chem.MolFromSmiles(entry[0] if isinstance(entry, (list, tuple)) else entry) is not None
            for entry in smiles]


def _v2_restore_invalid(values, mask):
    """Re-expands a list computed over the valid entries back to the full query."""
    it = iter(values)
    return [next(it) if ok else None for ok in mask]


def _v2_conformal_halfwidths(extras, task_names):
    """Half-interval widths from a conformal-calibrated chemprop 2.x prediction.

    chemprop 2.x reports a conformal interval as explicit lower/upper bound columns,
    while the rest of this app works in chemprop 1.x's shape of one half-width per
    task (the interval being prediction +/- half-width), so convert to that.
    """
    # A conformal-calibrated run reports the calibrated half interval in the same
    # uncertainty column an uncalibrated run uses, which is already the shape needed.
    halfwidths = backends.column_group(extras, task_names, backends.UNCERTAINTY_SUFFIXES)
    if halfwidths is not None:
        return halfwidths

    lower = backends.column_group(extras, task_names, backends.INTERVAL_LOWER_SUFFIXES)
    upper = backends.column_group(extras, task_names, backends.INTERVAL_UPPER_SUFFIXES)
    if lower is None or upper is None:
        return None

    halfwidths = []
    for lo_row, up_row in zip(lower, upper):
        if lo_row is None or up_row is None:
            halfwidths.append(None)
            continue
        halfwidths.append([None if (lo is None or up is None) else (up - lo) / 2.0
                           for lo, up in zip(lo_row, up_row)])
    return halfwidths


def load_calibration_data(cal_path, task_names):
    """Read a saved conformal calibration CSV (``smiles`` + task columns) into a
    list-of-lists SMILES (for make_predictions) and per-task integer labels."""
    smiles, labels = [], []
    with open(cal_path) as f:
        for row in csv.DictReader(f):
            smiles.append([row.get('smiles', '')])
            lab = []
            for tn in task_names:
                v = (row.get(tn) or '').strip()
                try:
                    lab.append(int(round(float(v))) if v != '' else None)
                except ValueError:
                    lab.append(None)
            labels.append(lab)
    return smiles, labels


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that understands numpy scalar/array types."""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _results_path(ckpt_id) -> str:
    return os.path.join(app.config['CHECKPOINT_FOLDER'], f'{ckpt_id}_results.json')


def _meta_path(ckpt_id) -> str:
    return os.path.join(app.config['CHECKPOINT_FOLDER'], f'{ckpt_id}_meta.json')


def write_ckpt_meta(ckpt_id, **meta) -> None:
    """Records what predicting from a checkpoint needs to know about it.

    chemprop 1.x reads this from the checkpoint itself (``load_args`` /
    ``load_task_names``), but a chemprop 2.x checkpoint can only be opened by
    chemprop 2.x, and this process runs 1.x. Writing the same facts to a sidecar
    at training time keeps the predict path free of any v2 import.
    """
    with open(_meta_path(ckpt_id), 'w') as f:
        json.dump(meta, f)


def load_ckpt_meta(ckpt_id) -> Optional[dict]:
    """The metadata sidecar for a checkpoint, or None if it has none (v1 uploads)."""
    path = _meta_path(ckpt_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def ckpt_backend(ckpt_id) -> str:
    """Which chemprop version trained a checkpoint: 'v1' or 'v2'."""
    row = db.query_db('SELECT backend FROM ckpt WHERE id = ?', (ckpt_id,), one=True)
    return (row['backend'] if row and row['backend'] else 'v1')


def v2_train_dir(ckpt_id) -> str:
    """Working directory for a chemprop 2.x training run.

    Unlike the v1 backend's TemporaryDirectory, this outlives the request: the
    saved data splits are read afterwards by the background visualization thread,
    which removes the directory when it is done.
    """
    return os.path.join(app.config['TEMP_FOLDER'], f'v2_train_{ckpt_id}')


def _write_results_json(ckpt_id, data: dict) -> None:
    """Persist a checkpoint's results JSON (used by the Checkpoints page and the
    Train page's deferred-results poll). ``viz_status`` is one of pending/done/error."""
    with open(_results_path(ckpt_id), 'w') as f:
        json.dump(data, f, cls=_NumpyEncoder)


def _compute_train_visualization(ckpt_id, model_paths, data_path, id_col, ignore_cols,
                                 dataset_type, args, conformal_enabled, conformal_alpha,
                                 val_curves, result_key, backend='v1', v2_dir=None, gpu=None):
    """Heavy post-training work, run in a background thread so the Train request can
    return at once. Predicts the train/test splits to build the scatter/ROC plot data
    and stats, writes the per-checkpoint train/test predictions CSV, computes the
    conformal calibration set + coverage report, and finally writes the results JSON
    with ``viz_status='done'`` (or ``'error'``). The page polls the results endpoint.
    """
    warnings = []
    plot_data = None
    conformal_info = None
    viz_status = 'done'
    try:
        if backend == 'v2':
            # chemprop 2.x saved the splits it actually trained on. Reading those back
            # is both cheaper and safer than recreating the split here, since the two
            # versions do not partition a dataset identically.
            splits = []
            for split_name in ('train', 'val', 'test'):
                split_path = backends.find_split_file(v2_dir, split_name)
                if split_path is None:
                    raise backends.BackendError(f'chemprop 2 saved no {split_name} split')
                splits.append(get_data(path=split_path, smiles_columns=args.smiles_columns,
                                       ignore_columns=ignore_cols, store_row=bool(id_col)))
            train_split, val_split, test_split = splits
        else:
            # Reload data fresh from CSV — run_training scales targets in-place,
            # so reusing training data here would give scaled targets vs unscaled preds.
            fresh_data = get_data(path=data_path, smiles_columns=args.smiles_columns,
                                  ignore_columns=ignore_cols, store_row=bool(id_col))
            train_split, val_split, test_split = split_data(
                data=fresh_data,
                split_type=args.split_type,
                sizes=args.split_sizes,
                key_molecule_index=args.split_key_molecule,
                seed=args.seed,
                num_folds=args.num_folds,
                args=args,
            )

        _use_unc = len(model_paths) > 1

        if backend == 'v2':
            preloaded = None
            _v2_env = backends.subprocess_env(gpu)
            _v2_accelerator = backends.accelerator_for(gpu)

            def _v2_predict(dataset, uncertainty_method=None,
                            cal_path=None, calibration_method=None):
                """Predicts a split with the chemprop 2.x CLI; returns (preds, extras)."""
                query_path = os.path.join(v2_dir, 'viz_query.csv')
                preds_path = os.path.join(v2_dir, 'viz_preds.csv')
                query_smiles = to_smiles_list(dataset)
                mask = _v2_valid_mask(query_smiles)
                backends.write_smiles_csv(
                    [s for s, ok in zip(query_smiles, mask) if ok], query_path)
                if not any(mask):
                    return [None] * len(mask), {}

                cmd = backends.build_predict_cmd(
                    app.config['CHEMPROP2_BIN'], test_path=query_path, preds_path=preds_path, model_paths=model_paths,
                    uncertainty_method=uncertainty_method, cal_path=cal_path,
                    calibration_method=calibration_method,
                    conformal_alpha=conformal_alpha if calibration_method else None,
                    accelerator=_v2_accelerator)

                proc = backends.run_cli(cmd, os.path.join(v2_dir, 'predict.log'), _v2_env)
                proc.join()
                if proc.exitcode != 0:
                    raise backends.BackendError(
                        f'chemprop 2 prediction failed (exit code {proc.exitcode})')

                preds, extras = backends.read_preds(preds_path, args.task_names)
                return (_v2_restore_invalid(preds, mask),
                        {name: _v2_restore_invalid(values, mask)
                         for name, values in extras.items()})

            def _predict(dataset):
                preds, extras = _v2_predict(dataset, 'ensemble' if _use_unc else None)
                if not _use_unc:
                    return preds
                return preds, backends.column_group(extras, args.task_names,
                                                    backends.UNCERTAINTY_SUFFIXES)

            def _predict_plain(dataset):
                return _v2_predict(dataset)[0]

            pred_args = None
        else:
            pred_arguments = [
                '--test_path', 'None',
                '--preds_path', os.path.join(app.config['TEMP_FOLDER'], 'train_plot_preds.csv'),
                '--checkpoint_paths', *model_paths,
                # This runs in a background thread; chemprop's default of 8 DataLoader
                # worker *processes* cannot be spawned reliably from a non-main thread
                # (BrokenPipe/hangs), so load data inline.
                '--num_workers', '0',
            ]
            if _use_unc:
                pred_arguments += ['--uncertainty_method', 'ensemble']
            if not args.cuda:
                pred_arguments.append('--no_cuda')
            elif hasattr(args, 'gpu') and args.gpu is not None:
                pred_arguments += ['--gpu', str(args.gpu)]
            if args.features_generator is not None:
                pred_arguments += ['--features_generator', *args.features_generator]
                if not args.features_scaling:
                    pred_arguments.append('--no_features_scaling')

            pred_args = PredictArgs().parse_args(pred_arguments)

            # Load the ensemble once and reuse it across the train/test passes, so the
            # models aren't re-read from disk for every split.
            from chemprop.train.make_predictions import load_model
            preloaded = load_model(pred_args, generator=False)

            def _predict(dataset):
                return make_predictions(args=pred_args, smiles=to_smiles_list(dataset),
                                        model_objects=preloaded, return_uncertainty=_use_unc)

            def _predict_plain(dataset):
                return make_predictions(args=pred_args, smiles=to_smiles_list(dataset),
                                        model_objects=preloaded, return_uncertainty=False)

        def _tt_var_to_std(row):
            if row is None:
                return None
            result = []
            for v in row:
                try:
                    fv = float(v)
                    result.append(round(math.sqrt(fv), 3) if fv >= 0 else None)
                except (TypeError, ValueError):
                    result.append(None)
            return result

        def to_smiles_list(dataset):
            smiles = dataset.smiles()
            if smiles and isinstance(smiles[0], str):
                return [[s] for s in smiles]
            return smiles

        def flat_smiles(dataset):
            smiles = dataset.smiles()
            if smiles and isinstance(smiles[0], str):
                return smiles
            return [s[0] for s in smiles]

        train_ids = [d.row.get(id_col, '') for d in train_split] if id_col else None
        test_ids  = [d.row.get(id_col, '') for d in test_split]  if id_col else None

        tt_path = os.path.join(app.config['CHECKPOINT_FOLDER'], f'{ckpt_id}_train_test_preds.csv')

        if dataset_type == 'regression':
            if _use_unc:
                train_preds, train_unc = _predict(train_split)
                test_preds, test_unc = _predict(test_split)
                train_std = [_tt_var_to_std(r) for r in train_unc]
                test_std  = [_tt_var_to_std(r) for r in test_unc]
            else:
                train_preds = _predict(train_split)
                test_preds = _predict(test_split)
                train_std = test_std = None
            train_targets = train_split.targets()
            test_targets = test_split.targets()
            train_smiles = flat_smiles(train_split)
            test_smiles = flat_smiles(test_split)

            def _reg_stats(pts):
                if not pts:
                    return None
                y = np.array([p[0] for p in pts], dtype=float)
                p = np.array([p[1] for p in pts], dtype=float)
                ss_res = np.sum((y - p) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
                rmse = float(np.sqrt(np.mean((y - p) ** 2)))
                mae = float(np.mean(np.abs(y - p)))
                return {'r2': f'{r2:.3f}', 'rmse': f'{rmse:.3f}', 'mae': f'{mae:.3f}', 'n': len(pts)}

            plot_data = []
            for i, task_name in enumerate(args.task_names):
                train_pts = [[train_targets[j][i], train_preds[j][i], train_smiles[j]]
                             for j in range(len(train_preds))
                             if train_preds[j] is not None and train_targets[j][i] is not None]
                test_pts = [[test_targets[j][i], test_preds[j][i], test_smiles[j]]
                            for j in range(len(test_preds))
                            if test_preds[j] is not None and test_targets[j][i] is not None]
                plot_data.append({
                    'name': task_name,
                    'train': train_pts,
                    'test': test_pts,
                    'train_stats': _reg_stats(train_pts),
                    'test_stats': _reg_stats(test_pts),
                })

            with open(tt_path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = (['id'] if id_col else []) + ['smiles', 'split']
                for name in args.task_names:
                    header += [name, f'pred_{name}']
                if train_std is not None:
                    for name in args.task_names:
                        header.append(f'std_{name}')
                writer.writerow(header)
                for split_label, smi_list, tgts, preds_list, std_list, ids in [
                    ('train', train_smiles, train_targets, train_preds, train_std, train_ids),
                    ('test',  test_smiles,  test_targets,  test_preds,  test_std,  test_ids),
                ]:
                    for j in range(len(preds_list)):
                        if preds_list[j] is None:
                            continue
                        row = ([ids[j]] if id_col else []) + [smi_list[j], split_label]
                        for i in range(len(args.task_names)):
                            t_val = tgts[j][i]
                            p_val = preds_list[j][i]
                            row += [
                                round(t_val, 3) if t_val is not None else '',
                                round(p_val, 3) if p_val is not None else '',
                            ]
                        if std_list is not None:
                            srow = std_list[j] if std_list[j] is not None else [None] * len(args.task_names)
                            row += [v if v is not None else '' for v in srow]
                        writer.writerow(row)

        elif dataset_type == 'classification':
            from sklearn.metrics import (roc_curve, auc as sklearn_auc,
                                         accuracy_score, precision_score,
                                         recall_score, f1_score,
                                         matthews_corrcoef, confusion_matrix)
            if _use_unc:
                train_preds, train_unc = _predict(train_split)
                test_preds, test_unc = _predict(test_split)
                train_std = [_tt_var_to_std(r) for r in train_unc]
                test_std  = [_tt_var_to_std(r) for r in test_unc]
            else:
                train_preds = _predict(train_split)
                test_preds = _predict(test_split)
                train_std = test_std = None
            train_targets = train_split.targets()
            test_targets = test_split.targets()
            train_smiles = flat_smiles(train_split)
            test_smiles = flat_smiles(test_split)

            plot_data = []
            for i, task_name in enumerate(args.task_names):
                t = [test_targets[j][i] for j in range(len(test_preds))
                     if test_preds[j] is not None and test_targets[j][i] is not None]
                p = [test_preds[j][i] for j in range(len(test_preds))
                     if test_preds[j] is not None and test_targets[j][i] is not None]
                if len(set(t)) == 2:
                    fpr, tpr, _ = roc_curve(t, p)
                    roc_auc = sklearn_auc(fpr, tpr)
                    p_bin = [1 if prob >= 0.5 else 0 for prob in p]
                    tn, fp, fn, tp = confusion_matrix(t, p_bin).ravel()
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    test_stats = {
                        'n': len(t),
                        'auc': round(roc_auc, 3),
                        'accuracy': round(float(accuracy_score(t, p_bin)), 3),
                        'precision': round(float(precision_score(t, p_bin, zero_division=0)), 3),
                        'recall': round(float(recall_score(t, p_bin, zero_division=0)), 3),
                        'specificity': round(float(specificity), 3),
                        'f1': round(float(f1_score(t, p_bin, zero_division=0)), 3),
                        'mcc': round(float(matthews_corrcoef(t, p_bin)), 3),
                        'tn': int(tn), 'fp': int(fp),
                        'fn': int(fn), 'tp': int(tp),
                        'n_pos': int(fn + tp),
                        'n_neg': int(tn + fp),
                    }
                    plot_data.append({
                        'name': task_name,
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist(),
                        'auc': round(roc_auc, 3),
                        'test_stats': test_stats,
                    })

            with open(tt_path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = (['id'] if id_col else []) + ['smiles', 'split']
                for name in args.task_names:
                    header += [name, f'pred_prob_{name}']
                if train_std is not None:
                    for name in args.task_names:
                        header.append(f'std_{name}')
                writer.writerow(header)
                for split_label, smi_list, tgts, preds_list, std_list, ids in [
                    ('train', train_smiles, train_targets, train_preds, train_std, train_ids),
                    ('test',  test_smiles,  test_targets,  test_preds,  test_std,  test_ids),
                ]:
                    for j in range(len(preds_list)):
                        if preds_list[j] is None:
                            continue
                        row = ([ids[j]] if id_col else []) + [smi_list[j], split_label]
                        for i in range(len(args.task_names)):
                            t_val = tgts[j][i]
                            p_val = preds_list[j][i]
                            row += [
                                int(t_val) if t_val is not None else '',
                                round(p_val, 3) if p_val is not None else '',
                            ]
                        if std_list is not None:
                            srow = std_list[j] if std_list[j] is not None else [None] * len(args.task_names)
                            row += [v if v is not None else '' for v in srow]
                        writer.writerow(row)

        # Conformal: save the held-out validation split as the calibration set and
        # report empirical coverage on the independent test split.
        if conformal_enabled:
            n_cal = len(val_split)
            if not conformal_calibration_feasible(n_cal, conformal_alpha):
                warnings.append(
                    f'Conformal prediction skipped: the validation split (n={n_cal}) is too '
                    f'small to calibrate at alpha={conformal_alpha}. Train on more data or '
                    f'increase alpha.'
                )
            else:
                cal_path = conformal_calibration_path(ckpt_id)
                val_smiles = flat_smiles(val_split)
                val_targets = val_split.targets()
                with open(cal_path, 'w', newline='') as f:
                    cwriter = csv.writer(f)
                    cwriter.writerow(['smiles', *args.task_names])
                    for s, t in zip(val_smiles, val_targets):
                        cwriter.writerow([s, *['' if v is None else v for v in t]])

                if dataset_type == 'regression':
                    if backend == 'v2':
                        conf_preds, conf_extras = _v2_predict(
                            test_split,
                            uncertainty_method=backends.conformal_uncertainty_method(
                                len(model_paths)),
                            cal_path=cal_path, calibration_method='conformal-regression')
                        conf_unc = _v2_conformal_halfwidths(conf_extras, args.task_names)
                        if conf_unc is None:
                            raise backends.BackendError(
                                'chemprop 2 returned no conformal interval columns')
                    else:
                        conf_arguments = [
                            '--test_path', 'None',
                            '--preds_path', os.path.join(app.config['TEMP_FOLDER'], 'train_conformal_preds.csv'),
                            '--checkpoint_paths', *model_paths,
                            '--smiles_columns', 'smiles',
                            '--calibration_path', cal_path,
                            '--calibration_method', 'conformal_regression',
                            '--conformal_alpha', str(conformal_alpha),
                            # See note above: avoid worker-process DataLoader in this thread.
                            '--num_workers', '0',
                        ]
                        if not args.cuda:
                            conf_arguments.append('--no_cuda')
                        elif hasattr(args, 'gpu') and args.gpu is not None:
                            conf_arguments += ['--gpu', str(args.gpu)]
                        if args.features_generator is not None:
                            conf_arguments += ['--features_generator', *args.features_generator]
                            if not args.features_scaling:
                                conf_arguments.append('--no_features_scaling')

                        conf_args = PredictArgs().parse_args(conf_arguments)
                        conf_preds, conf_unc = make_predictions(
                            args=conf_args, smiles=to_smiles_list(test_split), return_uncertainty=True)
                    test_targets_c = test_split.targets()
                    per_task = []
                    for ti, name in enumerate(args.task_names):
                        covered = total = 0
                        widths = []
                        for j in range(len(conf_preds)):
                            yt = test_targets_c[j][ti]
                            if yt is None:
                                continue
                            try:
                                mid = float(conf_preds[j][ti])
                                half = float(conf_unc[j][ti])
                            except (TypeError, ValueError, IndexError):
                                continue
                            total += 1
                            if mid - half <= yt <= mid + half:
                                covered += 1
                            widths.append(2 * half)
                        per_task.append({
                            'name': name,
                            'coverage': round(covered / total, 3) if total else None,
                            'mean_width': round(float(np.mean(widths)), 3) if widths else None,
                            'n': total,
                        })
                    conformal_info = {'enabled': True, 'alpha': conformal_alpha, 'n_cal': n_cal,
                                      'mode': 'regression', 'per_task': per_task}
                else:
                    # Classification: Mondrian (class-conditional) conformal. Calibrate a
                    # per-class threshold on the validation set's plain probabilities, then
                    # measure per-class coverage on the independent test split, reusing the
                    # probabilities already predicted for the ROC.
                    val_preds = _predict_plain(val_split)
                    thr = mondrian_conformal_thresholds(val_preds, val_split.targets(), conformal_alpha)
                    test_targets_c = test_split.targets()
                    per_task = []
                    for ti, name in enumerate(args.task_names):
                        t = thr[ti] if ti < len(thr) else {'q_active': None, 'q_inactive': None}
                        cov_a = cov_i = na = ni = 0
                        cats = {'active': 0, 'inactive': 0, 'uncertain': 0, 'none': 0}
                        for j in range(len(test_preds)):
                            if test_preds[j] is None:
                                continue
                            try:
                                p = float(test_preds[j][ti])
                            except (TypeError, ValueError, IndexError):
                                continue
                            cat = mondrian_category(p, t)
                            if cat:
                                cats[cat] += 1
                            yt = test_targets_c[j][ti]
                            if yt is None:
                                continue
                            active_in = t['q_active'] is not None and (1.0 - p) <= t['q_active']
                            inactive_in = t['q_inactive'] is not None and p <= t['q_inactive']
                            if int(round(float(yt))) == 1:
                                na += 1; cov_a += active_in
                            else:
                                ni += 1; cov_i += inactive_in
                        per_task.append({
                            'name': name,
                            'active_coverage': round(cov_a / na, 3) if na else None,
                            'inactive_coverage': round(cov_i / ni, 3) if ni else None,
                            'n_active': na, 'n_inactive': ni,
                            'categories': cats,
                        })
                    conformal_info = {'enabled': True, 'alpha': conformal_alpha, 'n_cal': n_cal,
                                      'mode': 'classification', 'method': 'mondrian', 'per_task': per_task}

    except Exception as e:
        viz_status = 'error'
        plot_data = None
        conformal_info = None
        warnings.append(f'Could not generate visualization: {e}')

    # Release model weights and molecule-graph caches held by this thread.
    # SMILES_TO_GRAPH / SMILES_TO_MOL are process-global and never auto-cleared;
    # without this they accumulate across every training job and cause the server
    # to grow to several GB after a few runs.
    try:
        del preloaded
    except NameError:
        pass
    empty_cache()

    # The v2 run directory is kept alive past the training request only so the saved
    # splits can be read here; nothing needs it now.
    if backend == 'v2' and v2_dir:
        shutil.rmtree(v2_dir, ignore_errors=True)

    _write_results_json(ckpt_id, {'dataset_type': dataset_type, 'plot_data': plot_data,
                                  'val_curves': val_curves, 'conformal': conformal_info,
                                  'viz_status': viz_status, 'warnings': warnings})

    # Refresh the in-memory snapshot so the page reload (and any tab switch) renders
    # the finished plots/conformal via the normal server-side template.
    last = LAST_TRAIN_RESULT.get(result_key)
    if last is not None and last.get('ckpt_id') == ckpt_id:
        last.update(plot_data=plot_data, conformal=conformal_info, viz_pending=False,
                    warnings=(last.get('warnings') or []) + warnings)


@app.context_processor
def inject_version():
    return {'web_version': app.config['WEB_VERSION']}


def is_admin() -> bool:
    """Whether the logged-in user has admin rights (e.g. creating users)."""
    username = (session.get('username') or '').lower()
    return username in app.config.get('ADMIN_USERS', [])


@app.context_processor
def inject_is_admin():
    """Exposes ``is_admin`` to templates so admin-only UI can be hidden."""
    return {'is_admin': is_admin()}


def check_not_demo(func: Callable) -> Callable:
    """
    View wrapper, which will redirect request to site
    homepage if app is run in DEMO mode.
    :param func: A view which performs sensitive behavior.
    :return: A view with behavior adjusted based on DEMO flag.
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        if app.config['DEMO']:
            return redirect(url_for('home'))
        return func(*args, **kwargs)

    return decorated_function


def check_admin(func: Callable) -> Callable:
    """View wrapper that redirects non-admin users to the homepage."""
    @wraps(func)
    def decorated_function(*args, **kwargs):
        if not is_admin():
            return redirect(url_for('home'))
        return func(*args, **kwargs)

    return decorated_function


def progress_bar(args: TrainArgs, progress: mp.Value):
    """
    Updates a progress bar displayed during training.

    :param args: Arguments.
    :param progress: The current progress.
    """
    total_epochs = args.epochs * args.ensemble_size
    log_path = os.path.join(args.save_dir, 'verbose.log')
    while progress.value < 100:
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                epochs_done = f.read().count('Epoch ')
            progress.value = min(epochs_done * 100 / total_epochs, 100)
        time.sleep(0.5)


def progress_bar_v2(output_dir: str, total_epochs: int, progress: mp.Value):
    """Progress for a chemprop 2.x run, read from its Lightning metrics files.

    The v1 progress bar counts ``Epoch`` lines in chemprop's own log; chemprop 2.x
    logs through Lightning instead, so progress comes from the metrics CSVs.

    :param output_dir: The v2 run's output directory.
    :param total_epochs: Epochs times ensemble size.
    :param progress: The current progress.
    """
    while progress.value < 100:
        progress.value = backends.epoch_progress(output_dir, total_epochs)
        time.sleep(0.5)


def find_unused_path(path: str) -> str:
    """
    Given an initial path, finds an unused path by appending different numbers to the filename.

    :param path: An initial path.
    :return: An unused path.
    """
    if not os.path.exists(path):
        return path

    base_name, ext = os.path.splitext(path)

    i = 2
    while os.path.exists(path):
        path = base_name + str(i) + ext
        i += 1

    return path


def name_already_exists_message(thing_being_named: str, original_name: str, new_name: str) -> str:
    """
    Creates a message about a path already existing and therefore being renamed.

    :param thing_being_named: The thing being renamed (ex. Data, Checkpoint).
    :param original_name: The original name of the object.
    :param new_name: The new name of the object.
    :return: A string with a message about the changed name.
    """
    return f'{thing_being_named} "{original_name} already exists. ' \
           f'Saving to "{new_name}".'


def get_upload_warnings_errors(upload_item: str) -> Tuple[List[str], List[str]]:
    """
    Gets any upload warnings passed along in the request.

    :param upload_item: The thing being uploaded (ex. Data, Checkpoint).
    :return: A tuple with a list of warning messages and a list of error messages.
    """
    warnings_raw = request.args.get(f'{upload_item}_upload_warnings')
    errors_raw = request.args.get(f'{upload_item}_upload_errors')
    warnings = json.loads(warnings_raw) if warnings_raw is not None else None
    errors = json.loads(errors_raw) if errors_raw is not None else None

    return warnings, errors


def format_float(value: float, precision: int = 4) -> str:
    """
    Formats a float value to a specific precision.

    :param value: The float value to format.
    :param precision: The number of decimal places to use.
    :return: A string containing the formatted float.
    """
    return f'{value:.{precision}f}'


def format_float_list(array: List[float], precision: int = 4) -> List[str]:
    """
    Formats a list of float values to a specific precision.

    :param array: A list of float values to format.
    :param precision: The number of decimal places to use.
    :return: A list of strings containing the formatted floats.
    """
    return [format_float(f, precision) for f in array]


@app.teardown_request
def release_job_after_error(exception):
    """Drops a job whose request died, so the page does not poll a run that is over.

    Every view removes its own job on the way out, but an unhandled exception
    skips that, leaving /receiver reporting a job that will never finish.
    """
    if exception is None:
        return
    try:
        job = current_job()
        if job is not None and not job.is_running():
            end_job(job)
    except Exception:
        pass


@app.route('/receiver', methods=['POST'])
@check_not_demo
def receiver():
    """Receiver monitoring the progress of training."""
    job = current_job()
    if job is None:
        return jsonify(progress=0.0, training=0, mode='', val_curves={})

    val_curves = {}
    if job.v2_dir:
        val_curves = backends.val_curves(job.v2_dir)
    elif job.log_path and os.path.exists(job.log_path):
        val_curves = _parse_val_curves(job.log_path)

    return jsonify(progress=job.progress.value, training=1,
                   mode=job.mode, val_curves=val_curves)


@app.route('/cancel', methods=['POST'])
def cancel():
    """Terminates the current user's training, hyperopt or prediction job."""
    job = current_job()
    if job is not None:
        job.stop()
        job.progress.value = 0.0
    return jsonify(success=True)


# Endpoints reachable without being logged in.
PUBLIC_ENDPOINTS = {'login', 'static'}


def current_user_id() -> Optional[int]:
    """The authenticated user's id, read from the signed session.

    This replaces the legacy ``currentUser`` cookie as the source of identity,
    so a client can no longer reach another user's data by editing a cookie.
    """
    return session.get('user_id')


def owned_ckpt(ckpt_id):
    """The current user's checkpoint row, or None when it is not theirs.

    Checkpoint ids are small sequential integers that appear in URLs and forms, so
    every route taking one has to look it up scoped to the requester rather than
    trust the id it was handed.
    """
    try:
        ckpt_id = int(ckpt_id)
    except (TypeError, ValueError):
        return None

    return db.query_db('SELECT * FROM ckpt WHERE id = ? AND associated_user = ?',
                       (ckpt_id, current_user_id() or app.config['DEFAULT_USER_ID']),
                       one=True)


def owned_dataset(dataset_id):
    """The current user's dataset row, or None when it is not theirs."""
    try:
        dataset_id = int(dataset_id)
    except (TypeError, ValueError):
        return None

    return db.query_db('SELECT * FROM dataset WHERE id = ? AND associated_user = ?',
                       (dataset_id, current_user_id() or app.config['DEFAULT_USER_ID']),
                       one=True)


@app.before_request
def require_login():
    """Redirects unauthenticated requests to the login page.

    Skipped in DEMO mode, which exposes no per-user accounts, and for the
    login page and static assets.
    """
    if app.config.get('DEMO'):
        return None

    if request.endpoint in PUBLIC_ENDPOINTS or session.get('user_id') is not None:
        return None

    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Authenticates a user against the file-based credential store."""
    if app.config.get('DEMO'):
        return redirect(url_for('home'))

    if request.method == 'GET':
        return render_template('login.html')

    username = request.form.get('username', '').strip()
    password = request.form.get('password', '')

    if auth.verify_password(username, password):
        session.clear()
        session.permanent = True
        session['user_id'] = db.get_or_create_user(username)
        session['username'] = username
        return redirect(url_for('home'))

    return render_template('login.html', error='Invalid username or password.')


@app.route('/logout')
def logout():
    """Logs the current user out by clearing their session."""
    session.clear()
    return redirect(url_for('login'))


@app.route('/')
def home():
    """Renders the home page."""
    return render_template('home.html', users=db.get_all_users())


@app.route('/create_user', methods=['GET', 'POST'])
@check_not_demo
@check_admin
def create_user():
    """
    If a POST request is made, creates a new user.
    Renders the create_user page.
    """
    if request.method == 'GET':
        return render_template('create_user.html', users=db.get_all_users())

    new_name = request.form.get('newUserName', '').strip()
    password = request.form.get('password', '')

    if not new_name or not password:
        return render_template('create_user.html', users=db.get_all_users(),
                               error='Both a username and a password are required.')

    if auth.user_exists(new_name):
        return render_template('create_user.html', users=db.get_all_users(),
                               error=f'A user named "{new_name}" already exists.')

    auth.set_password(new_name, password)
    db.get_or_create_user(new_name)

    return render_template('create_user.html', users=db.get_all_users(),
                           message=f'Created user "{new_name}".')


def render_train(**kwargs):
    """Renders the train page with specified kwargs."""
    data_upload_warnings, data_upload_errors = get_upload_warnings_errors('data')

    # On GET with no explicit result, serve the last training result (e.g. user switched tabs mid-training)
    if request.method == 'GET' and 'trained' not in kwargs:
        result_key = str(current_user_id() or '')
        if result_key in LAST_TRAIN_RESULT:
            last = dict(LAST_TRAIN_RESULT[result_key])
            _ckpt_id = last.get('ckpt_id')
            _perm = os.path.join(app.config['CHECKPOINT_FOLDER'], f'{_ckpt_id}_train_test_preds.csv') if _ckpt_id else ''
            last['train_test_preds_available'] = bool(_ckpt_id) and os.path.exists(_perm)
            kwargs = {**last, **kwargs}

    # Repopulate the form with the user's last-applied settings (until they submit new ones).
    kwargs.setdefault('settings', LAST_TRAIN_SETTINGS.get(str(current_user_id() or ''), {}))

    return render_template('train.html',
                           datasets=db.get_datasets(current_user_id()),
                           current_user=current_user_id() or '',
                           cuda=app.config['CUDA'],
                           gpus=app.config['GPUS'],
                           chemprop2_available=app.config['CHEMPROP2_AVAILABLE'],
                           foundation_models=app.config['FOUNDATION_MODELS'],
                           foundation_checkpoints=foundation_checkpoints(current_user_id()),
                           default_batch_size=app.config['CHEMPROP2_BATCH_SIZE'],
                           data_upload_warnings=data_upload_warnings,
                           data_upload_errors=data_upload_errors,
                           users=db.get_all_users(),
                           **kwargs)


@app.route('/train', methods=['GET', 'POST'])
@check_not_demo
def train():
    """Renders the train page and performs training if request method is POST."""
    warnings, errors = [], []

    if request.method == 'GET':
        return render_train()

    # One job per user at a time: they share a GPU and, until this was keyed by
    # user, a second run left the first one running invisibly with the progress
    # bar and the Cancel button both pointing at the newer job.
    running = current_job()
    if running is not None and running.is_running():
        errors.append(f'You already have a {running.mode} job running. Wait for it to '
                      f'finish, or cancel it, before starting another.')
        return render_train(warnings=warnings, errors=errors)

    # Remember the submitted form values so the page repopulates with them (instead of
    # resetting to defaults) until the user applies new settings. Raw strings are stored
    # so the template can re-select/re-check fields directly.
    LAST_TRAIN_SETTINGS[str(current_user_id() or '')] = {
        k: request.form.get(k, '') for k in (
            'dataName', 'idColumn', 'datasetType', 'splitType', 'featuresGenerator',
            'epochs', 'ensembleSize', 'patience', 'minDelta', 'seed',
            'conformalEnabled', 'conformalAlpha', 'checkpointName', 'gpu',
            'binarizeEnabled', 'binarizeMethod', 'binarizeParam',
            'backend', 'foundation', 'foundationEnabled', 'batchSize')
    }

    # Get arguments
    epochs, ensemble_size, checkpoint_name = \
        int(request.form['epochs']), int(request.form['ensembleSize']), \
        request.form['checkpointName']

    # The dataset id becomes a file path below, so it is resolved through the
    # registry scoped to this user rather than pasted into one.
    dataset_row = owned_dataset(request.form.get('dataName'))
    if dataset_row is None:
        errors.append('That dataset is not available.')
        return render_train(warnings=warnings, errors=errors)
    data_name = dataset_row['id']
    gpu = request.form.get('gpu')
    patience_raw = request.form.get('patience', '').strip()
    # Early stopping is gated by a checkbox on the form: when unchecked the
    # patience input is disabled and therefore not submitted, so no value here
    # means "disabled". A 0 is also treated as disabled defensively.
    patience = int(patience_raw) if patience_raw and int(patience_raw) > 0 else None
    # Minimum validation-metric improvement that counts as progress for early
    # stopping; smaller (noise-level) gains do not reset the patience counter.
    min_delta_raw = request.form.get('minDelta', '').strip()
    try:
        min_delta = float(min_delta_raw) if min_delta_raw else 0.0
    except ValueError:
        min_delta = 0.0
    if min_delta < 0:
        min_delta = 0.0
    data_path = os.path.join(app.config['DATA_FOLDER'], f'{data_name}.csv')
    dataset_type = request.form.get('datasetType', 'regression')
    split_type = request.form.get('splitType', 'random')
    if split_type not in ('random', 'scaffold_balanced'):
        split_type = 'random'
    id_col = request.form.get('idColumn', '').strip() or None
    if id_col and id_col not in get_header(data_path):
        warnings.append(f'Identifier column "{id_col}" not found in data — it will be ignored.')
        id_col = None
    ignore_cols = [id_col] if id_col else []
    features_generator = request.form.get('featuresGenerator', 'none')

    # Which chemprop trains this model. 'v2' runs the chemprop 2.x CLI in its own
    # conda environment, which is the only way to finetune a foundation model such
    # as CheMeleon: those checkpoints cannot be loaded by the 1.x code running here.
    backend = request.form.get('backend', 'v1')
    if backend not in ('v1', 'v2'):
        backend = 'v1'
    if backend == 'v2' and not app.config['CHEMPROP2_AVAILABLE']:
        errors.append(f'The chemprop 2 environment ("{app.config["CHEMPROP2_ENV"]}") was not '
                      f'found on this server, so foundation models are unavailable.')
        return render_train(warnings=warnings, errors=errors)

    # Finetuning from a foundation model is an option within the v2 backend, not the
    # reason for it: chemprop 2 also trains a plain D-MPNN from scratch.
    foundation_choice = request.form.get('foundation', '').strip() or None
    if backend != 'v2' or request.form.get('foundationEnabled', 'True') == 'False':
        foundation_choice = None
    try:
        foundation, foundation_label = resolve_foundation(foundation_choice)
    except ValueError as e:
        errors.append(str(e))
        return render_train(warnings=warnings, errors=errors)

    # Batch size is offered for the v2 backend only. chemprop 2 defaults to 64,
    # which leaves the GPU idling on datasets of this size; the v1 backend keeps
    # its own default so existing checkpoints stay reproducible.
    batch_size = app.config['CHEMPROP2_BATCH_SIZE']
    batch_raw = request.form.get('batchSize', '').strip()
    if batch_raw:
        try:
            batch_size = max(1, int(batch_raw))
        except ValueError:
            warnings.append(f'Ignoring invalid batch size "{batch_raw}".')

    use_progress_bar = request.form.get('useProgressBar', 'True') == 'True'
    conformal_enabled, conformal_alpha = parse_conformal_form(request.form)
    # Random seed controls both the train/val/test split and the initial model
    # weights, so a run is fully reproducible. Default 666.
    seed_raw = request.form.get('seed', '').strip()
    try:
        seed = int(seed_raw) if seed_raw else 666
    except ValueError:
        seed = 666

    # Auto-binarize settings (classification only)
    binarize_enabled = request.form.get('binarizeEnabled', 'False') == 'True'
    binarize_method = request.form.get('binarizeMethod', 'mad')
    if binarize_method not in ('mad', 'percentile', 'fixed'):
        binarize_method = 'mad'
    _default_param = {'mad': 3.0, 'percentile': 80.0, 'fixed': 0.0}
    binarize_param_raw = request.form.get('binarizeParam', '').strip()
    try:
        binarize_param = float(binarize_param_raw) if binarize_param_raw else _default_param[binarize_method]
    except ValueError:
        binarize_param = _default_param[binarize_method]

    if backend == 'v2':
        # Both are chemprop 1.x-only code paths on this server; the form hides them
        # when the v2 backend is selected, so this only catches a stale submission.
        if features_generator != 'none':
            warnings.append('Additional molecule-level features are not supported by the '
                            'chemprop 2 backend and were ignored.')
            features_generator = 'none'

    # Handle optional hyperopt config (content sent as hidden field via FileReader).
    # Both backends read their own hyperopt output: chemprop 1.x a JSON file,
    # chemprop 2 a config file for its own parser.
    config_content = request.form.get('configFileContent', '').strip()
    config_path = None
    if config_content:
        suffix = 'toml' if backend == 'v2' else 'json'
        config_path = os.path.join(app.config['TEMP_FOLDER'], f'uploaded_config.{suffix}')
        with open(config_path, 'w') as f:
            f.write(config_content)

    # Create and modify args
    train_arg_list = [
        '--data_path', data_path,
        '--dataset_type', dataset_type,
        '--epochs', str(epochs),
        '--ensemble_size', str(ensemble_size),
        '--split_type', split_type,
        '--seed', str(seed),
        '--pytorch_seed', str(seed),
    ]
    # chemprop 1.x reads its config as JSON; a chemprop 2 config is handed to the
    # v2 command instead, and would fail this parser.
    if config_path is not None and backend == 'v1':
        train_arg_list += ['--config_path', config_path]
    if gpu is not None:
        if gpu == 'None':
            train_arg_list.append('--no_cuda')
        else:
            train_arg_list += ['--gpu', gpu]
    if features_generator != 'none':
        train_arg_list += ['--features_generator', features_generator]
        if features_generator == 'rdkit_2d_normalized':
            train_arg_list.append('--no_features_scaling')
    args = TrainArgs().parse_args(train_arg_list)

    # Get task names
    args.task_names = get_task_names(path=data_path, smiles_columns=args.smiles_columns, ignore_columns=ignore_cols or None)

    # Safety catch: scan task columns for non-numeric values (likely an undeclared identifier column)
    non_numeric_cols = _find_non_numeric_columns(data_path, args.task_names)
    if non_numeric_cols:
        cols_str = ', '.join(f'"{c}"' for c in non_numeric_cols)
        errors.append(
            f'Column {cols_str} contains non-numeric values and cannot be used as a target. '
            f'If this is a compound identifier, enter its name in the "Identifier column" field above. '
            f'Otherwise remove it from your CSV before training.'
        )
        return render_train(warnings=warnings, errors=errors)

    # Check if regression/classification selection matches data
    try:
        data = get_data(path=data_path, smiles_columns=args.smiles_columns, ignore_columns=ignore_cols)
    except ValueError as e:
        errors.append(f'Failed to read training data: {e}')
        return render_train(warnings=warnings, errors=errors)

    targets = data.targets()
    unique_targets = {target for row in targets for target in row if target is not None}

    if dataset_type == 'classification' and len(unique_targets - {0, 1}) > 0:
        if binarize_enabled:
            binarize_out = os.path.join(app.config['TEMP_FOLDER'], f'binarized_{secure_filename(data_name)}.csv')
            try:
                binarize_stats = _binarize_csv(data_path, args.task_names, binarize_method, binarize_param, binarize_out)
            except Exception as e:
                errors.append(f'Auto-binarize failed: {e}')
                return render_train(warnings=warnings, errors=errors)

            degenerate = [s['name'] for s in binarize_stats
                          if s['n_total'] > 0 and (s['n_active'] == 0 or s['n_inactive'] == 0)]
            if degenerate:
                errors.append(
                    f'Auto-binarize produced all-one-class labels for: {", ".join(degenerate)}. '
                    f'Try a different threshold method or parameter value.')
                return render_train(warnings=warnings, errors=errors)

            # Redirect training to the binarized file
            data_path = binarize_out
            train_arg_list[train_arg_list.index('--data_path') + 1] = binarize_out

            _method_desc = {
                'mad': f'median + {binarize_param}×MAD',
                'percentile': f'≥ {binarize_param}th-percentile',
                'fixed': f'≥ {binarize_param}',
            }
            parts = []
            for s in binarize_stats:
                pct = round(100 * s['n_active'] / s['n_total'], 1) if s['n_total'] else 0
                parts.append(f"{s['name']}: threshold={s['threshold']}, "
                              f"{s['n_active']} active / {s['n_inactive']} inactive ({pct}%)")
            warnings.append(f"Auto-binarized ({_method_desc[binarize_method]}): {'; '.join(parts)}")
        else:
            errors.append(
                'Selected classification dataset but not all labels are 0 or 1. '
                'Enable auto-binarize above, or select regression instead.')
            return render_train(warnings=warnings, errors=errors)

    if dataset_type == 'regression' and unique_targets <= {0, 1}:
        errors.append('Selected regression dataset but all labels are 0 or 1. Select classification instead.')

        return render_train(warnings=warnings, errors=errors)

    current_user = current_user_id()

    if not current_user:
        # Use DEFAULT as current user if the client's cookie is not set.
        current_user = app.config['DEFAULT_USER_ID']

    ckpt_id, ckpt_name = db.insert_ckpt(checkpoint_name,
                                        current_user,
                                        args.dataset_type,
                                        args.epochs,
                                        args.ensemble_size,
                                        len(targets),
                                        backend=backend,
                                        foundation=foundation_label)

    # The v1 backend trains into a TemporaryDirectory that is gone by the time this
    # request returns. A v2 run must outlive it: the background visualization thread
    # reads the splits chemprop 2 saved there, and removes the directory when done.
    v2_dir = v2_train_dir(ckpt_id) if backend == 'v2' else None
    if v2_dir:
        shutil.rmtree(v2_dir, ignore_errors=True)

    with (nullcontext(v2_dir) if v2_dir else TemporaryDirectory()) as temp_dir:
        args.save_dir = temp_dir

        LAST_TRAIN_RESULT.pop(str(current_user), None)

        job = start_job('train')

        if use_progress_bar:
            if backend == 'v2':
                pb_proc = mp.Process(target=progress_bar_v2,
                                     args=(temp_dir, epochs * ensemble_size, job.progress))
            else:
                pb_proc = mp.Process(target=progress_bar, args=(args, job.progress))
            pb_proc.start()
            job.progress_bar = pb_proc

        job.log_path = os.path.join(temp_dir, 'verbose.log')
        if backend == 'v2':
            job.v2_dir = temp_dir
            train_cmd = backends.build_train_cmd(
                app.config['CHEMPROP2_BIN'], data_path=data_path, output_dir=temp_dir, task_type=dataset_type,
                task_names=args.task_names, smiles_column=get_header(data_path)[0],
                epochs=epochs, ensemble_size=ensemble_size, split_type=split_type,
                seed=seed, foundation=foundation, patience=patience, min_delta=min_delta,
                batch_size=batch_size, config_path=config_path,
                accelerator=backends.accelerator_for(gpu))
            train_proc = backends.run_cli(train_cmd, job.log_path,
                                          backends.subprocess_env(gpu))
        else:
            train_proc = _spawn.Process(target=_train_worker,
                                        args=(train_arg_list, args.task_names, data_path,
                                              ignore_cols, id_col, temp_dir, patience, min_delta))
        train_proc.start()
        job.process = train_proc

        while train_proc.is_alive():
            train_proc.join(timeout=0.5)

        job.process = None
        log_path = job.log_path
        cancelled = job.cancelled

        # The exit code cannot be used to recognise a cancellation: Lightning traps
        # SIGTERM and exits cleanly, so a killed chemprop 2 run returns 0. The job's
        # own flag is set only by this user's /cancel, which the page offers only
        # while the run is in progress, so it is the reliable signal.
        was_killed = cancelled

        if was_killed or train_proc.exitcode not in (0, None):
            if use_progress_bar and pb_proc.is_alive():
                pb_proc.terminate()
                pb_proc.join(timeout=2)
            end_job(job)
            if was_killed:
                if v2_dir:
                    shutil.rmtree(v2_dir, ignore_errors=True)
                return render_train(warnings=['Training was cancelled.'])
            # A failed v2 run keeps its directory: the CLI's output is the only
            # record of why it failed, and nothing else in this process saw it.
            errors.append(f'Training failed — see {os.path.join(v2_dir, "verbose.log")}.'
                          if v2_dir else
                          'Training failed — check server logs for details.')
            return render_train(warnings=warnings, errors=errors)

        if use_progress_bar:
            job.progress.value = 100
            pb_proc.join()
            job.progress_bar = None
            # The job stays registered until the results are fully prepared, so the
            # polling JS doesn't reload the page before LAST_TRAIN_RESULT is set.

        # Parse convergence data before temp_dir is cleaned up
        if backend == 'v2':
            val_curves = backends.val_curves(temp_dir)
        else:
            val_curves = _parse_val_curves(log_path)
        job.log_path = ''
        job.v2_dir = ''

        # Check if name overlap
        if checkpoint_name != ckpt_name:
            warnings.append(name_already_exists_message('Checkpoint', checkpoint_name, ckpt_name))

        # Move models
        if backend == 'v2':
            # A v2 run directory also holds Lightning checkpoints and the saved
            # splits, so take only the trained model of each ensemble member.
            trained_models = backends.collect_models(temp_dir)
            if not trained_models:
                shutil.rmtree(v2_dir, ignore_errors=True)
                end_job(job)
                errors.append('Training produced no model files — check server logs for details.')
                return render_train(warnings=warnings, errors=errors)
            for model_path in trained_models:
                model_id = db.insert_model(ckpt_id)
                shutil.move(model_path,
                            os.path.join(app.config['CHECKPOINT_FOLDER'], f'{model_id}.pt'))
        else:
            for root, _, files in os.walk(args.save_dir):
                for fname in files:
                    if fname.endswith('.pt'):
                        model_id = db.insert_model(ckpt_id)
                        save_path = os.path.join(app.config['CHECKPOINT_FOLDER'], f'{model_id}.pt')
                        shutil.move(os.path.join(args.save_dir, root, fname), save_path)

    # Everything the predict page needs to know about this checkpoint. For v2 models
    # it is the only source: their checkpoints can only be opened by chemprop 2.x.
    write_ckpt_meta(ckpt_id,
                    backend=backend,
                    foundation=foundation_label,
                    task_names=list(args.task_names),
                    dataset_type=dataset_type,
                    features_generator=None if features_generator == 'none' else features_generator,
                    smiles_column=get_header(data_path)[0])

    # Heavy visualization (predicting the train/test splits, conformal coverage) is
    # deferred to a background thread so this request returns immediately and the page
    # can show "training complete" while the plots are still being computed.
    model_paths = [os.path.join(app.config['CHECKPOINT_FOLDER'], f'{m["id"]}.pt')
                   for m in db.get_models(ckpt_id)]
    result_key = str(current_user)

    # Seed a pending results file + in-memory snapshot so the convergence chart and a
    # "generating plots" state can show right away.
    _write_results_json(ckpt_id, {'dataset_type': dataset_type, 'plot_data': None,
                                  'val_curves': val_curves, 'conformal': None,
                                  'viz_status': 'pending'})
    LAST_TRAIN_RESULT[result_key] = dict(
        trained=True, dataset_type=dataset_type, plot_data=None, ckpt_id=ckpt_id,
        val_curves=val_curves, conformal=None, viz_pending=True,
        warnings=list(warnings), errors=list(errors))

    if backend == 'v2':
        warm_v2_attribution(ckpt_id)

    viz_pending = dataset_type in ['regression', 'classification']
    if viz_pending:
        threading.Thread(
            target=_compute_train_visualization,
            args=(ckpt_id, model_paths, data_path, id_col, ignore_cols, dataset_type,
                  args, conformal_enabled, conformal_alpha, val_curves, result_key,
                  backend, v2_dir, gpu),
            daemon=True).start()
    else:
        # No scatter/ROC for other dataset types; mark results done immediately.
        _write_results_json(ckpt_id, {'dataset_type': dataset_type, 'plot_data': None,
                                      'val_curves': val_curves, 'conformal': None,
                                      'viz_status': 'done'})
        LAST_TRAIN_RESULT[result_key]['viz_pending'] = False
        if v2_dir:
            shutil.rmtree(v2_dir, ignore_errors=True)

    # Allow the polling JS to detect completion and reload to the results shell.
    end_job(job)

    return render_train(trained=True,
                        dataset_type=dataset_type,
                        plot_data=None,
                        ckpt_id=ckpt_id,
                        val_curves=val_curves,
                        conformal=None,
                        viz_pending=viz_pending,
                        train_test_preds_available=False,
                        warnings=warnings,
                        errors=errors)


def hyperopt_progress_bar(hyperopt_checkpoint_dir: str, num_iters: int, progress: mp.Value):
    """
    Updates a progress bar during hyperparameter optimization by counting completed trials.

    :param hyperopt_checkpoint_dir: Directory where completed trial .pkl files are written.
    :param num_iters: Total number of hyperopt trials.
    :param progress: Shared value tracking progress (0–100).
    """
    while progress.value < 100:
        if os.path.exists(hyperopt_checkpoint_dir):
            n_done = len([f for f in os.listdir(hyperopt_checkpoint_dir) if f.endswith('.pkl')])
            progress.value = min(n_done * 100 / num_iters, 99)
        time.sleep(1)


def hyperopt_progress_bar_v2(log_path: str, num_trials: int, progress: mp.Value):
    """Progress for a chemprop 2 hyperopt run, read from Ray Tune's status table."""
    while progress.value < 100:
        progress.value = backends.hpopt_progress(log_path, num_trials)
        time.sleep(1)


def render_hyperopt(**kwargs):
    """Renders the hyperopt page with specified kwargs."""
    data_upload_warnings, data_upload_errors = get_upload_warnings_errors('data')
    kwargs.setdefault('settings', LAST_HYPEROPT_SETTINGS.get(str(current_user_id() or ''), {}))
    return render_template('hyperopt.html',
                           chemprop2_available=app.config['CHEMPROP2_AVAILABLE'],
                           foundation_models=app.config['FOUNDATION_MODELS'],
                           foundation_checkpoints=foundation_checkpoints(current_user_id()),
                           datasets=db.get_datasets(current_user_id()),
                           cuda=app.config['CUDA'],
                           gpus=app.config['GPUS'],
                           data_upload_warnings=data_upload_warnings,
                           data_upload_errors=data_upload_errors,
                           users=db.get_all_users(),
                           **kwargs)


@app.route('/hyperopt', methods=['GET', 'POST'])
@check_not_demo
def hyperopt_page():
    """Renders the hyperopt page and runs hyperparameter optimization if request method is POST."""
    warnings, errors = [], []

    if request.method == 'GET':
        return render_hyperopt()

    running = current_job()
    if running is not None and running.is_running():
        errors.append(f'You already have a {running.mode} job running. Wait for it to '
                      f'finish, or cancel it, before starting another.')
        return render_hyperopt(warnings=warnings, errors=errors)

    # Remember submitted settings so the form repopulates with them until changed.
    LAST_HYPEROPT_SETTINGS[str(current_user_id() or '')] = {
        'dataName': request.form.get('dataName', ''),
        'idColumn': request.form.get('idColumn', ''),
        'datasetType': request.form.get('datasetType', 'regression'),
        'epochs': request.form.get('epochs', ''),
        'numIters': request.form.get('numIters', ''),
        'searchKeywords': request.form.getlist('searchKeywords'),
        'gpu': request.form.get('gpu', ''),
        'backend': request.form.get('backend', 'v1'),
        'foundation': request.form.get('foundation', ''),
        'foundationEnabled': request.form.get('foundationEnabled', 'True'),
    }

    # Get form fields
    dataset_row = owned_dataset(request.form.get('dataName'))
    if dataset_row is None:
        errors.append('That dataset is not available.')
        return render_hyperopt(warnings=warnings, errors=errors)
    data_name = dataset_row['id']
    epochs = int(request.form['epochs'])
    num_iters = int(request.form['numIters'])
    dataset_type = request.form.get('datasetType', 'regression')
    gpu = request.form.get('gpu')
    search_keywords = request.form.getlist('searchKeywords') or ['basic']
    id_col = request.form.get('idColumn', '').strip() or None

    backend = request.form.get('backend', 'v1')
    if backend not in ('v1', 'v2'):
        backend = 'v1'
    if backend == 'v2' and not app.config['CHEMPROP2_AVAILABLE']:
        errors.append(f'The chemprop 2 environment ("{app.config["CHEMPROP2_ENV"]}") was not '
                      f'found on this server.')
        return render_hyperopt(warnings=warnings, errors=errors)

    foundation_choice = request.form.get('foundation', '').strip() or None
    if backend != 'v2' or request.form.get('foundationEnabled', 'True') == 'False':
        foundation_choice = None
    try:
        foundation, _ = resolve_foundation(foundation_choice)
    except ValueError as e:
        errors.append(str(e))
        return render_hyperopt(warnings=warnings, errors=errors)

    data_path = os.path.join(app.config['DATA_FOLDER'], f'{data_name}.csv')

    if id_col and id_col not in get_header(data_path):
        warnings.append(f'Identifier column "{id_col}" not found in data — it will be ignored.')
        id_col = None
    ignore_cols = [id_col] if id_col else []

    # Safety catch: scan task columns for non-numeric values (likely an undeclared identifier column)
    task_names_preview = get_task_names(path=data_path, smiles_columns=None, ignore_columns=ignore_cols or None)
    non_numeric_cols = _find_non_numeric_columns(data_path, task_names_preview)
    if non_numeric_cols:
        cols_str = ', '.join(f'"{c}"' for c in non_numeric_cols)
        errors.append(
            f'Column {cols_str} contains non-numeric values and cannot be used as a target. '
            f'If this is a compound identifier, enter its name in the "Identifier column" field above. '
            f'Otherwise remove it from your CSV before hyperopt.'
        )
        return render_hyperopt(warnings=warnings, errors=errors)

    # Validate data type
    try:
        data = get_data(path=data_path, smiles_columns=None, ignore_columns=ignore_cols or None)
    except ValueError as e:
        errors.append(f'Failed to read training data: {e}')
        return render_hyperopt(warnings=warnings, errors=errors)
    targets = data.targets()
    unique_targets = {target for row in targets for target in row if target is not None}

    if dataset_type == 'classification' and len(unique_targets - {0, 1}) > 0:
        errors.append('Selected classification dataset but not all labels are 0 or 1. Select regression instead.')
        return render_hyperopt(warnings=warnings, errors=errors)

    if dataset_type == 'regression' and unique_targets <= {0, 1}:
        errors.append('Selected regression dataset but all labels are 0 or 1. Select classification instead.')
        return render_hyperopt(warnings=warnings, errors=errors)

    with TemporaryDirectory() as temp_dir:
        hyperopt_save_dir = os.path.join(temp_dir, 'hyperopt')
        os.makedirs(hyperopt_save_dir, exist_ok=True)
        config_save_path = os.path.join(app.config['TEMP_FOLDER'], 'hyperopt_config.json')

        hyper_args_list = [
            '--data_path', data_path,
            '--dataset_type', dataset_type,
            '--epochs', str(epochs),
            '--num_iters', str(num_iters),
            '--config_save_path', config_save_path,
            '--save_dir', hyperopt_save_dir,
            '--search_parameter_keywords', *search_keywords,
        ]
        if ignore_cols:
            hyper_args_list += ['--ignore_columns', *ignore_cols]

        if gpu is not None:
            if gpu == 'None':
                hyper_args_list.append('--no_cuda')
            else:
                hyper_args_list += ['--gpu', gpu]

        hyper_args = HyperoptArgs().parse_args(hyper_args_list)
        hyper_args.task_names = get_task_names(path=data_path, smiles_columns=hyper_args.smiles_columns)

        job = start_job('hyperopt')

        if backend == 'v2':
            # chemprop 2 writes its chosen settings as a config file for its own
            # parser, which the Train page can then feed back via --config-path.
            config_save_path = os.path.join(app.config['TEMP_FOLDER'], 'hyperopt_config.toml')
            # Outside the run's TemporaryDirectory: when a search fails, its output
            # is the only account of why, and it must outlive the request.
            log_path = os.path.join(app.config['TEMP_FOLDER'], 'hyperopt.log')
            job.log_path = log_path

            hpopt_cmd = backends.build_hpopt_cmd(
                app.config['CHEMPROP2_BIN'], data_path=data_path,
                save_dir=hyperopt_save_dir, task_type=dataset_type,
                task_names=hyper_args.task_names, smiles_column=get_header(data_path)[0],
                epochs=epochs, num_trials=num_iters, search_keywords=search_keywords,
                foundation=foundation, accelerator=backends.accelerator_for(gpu))

            pb_proc = mp.Process(target=hyperopt_progress_bar_v2,
                                 args=(log_path, num_iters, job.progress))
            hyper_proc = backends.run_cli(hpopt_cmd, log_path, backends.subprocess_env(gpu))
        else:
            pb_proc = mp.Process(target=hyperopt_progress_bar,
                                 args=(hyper_args.hyperopt_checkpoint_dir, num_iters, job.progress))
            hyper_proc = _spawn.Process(target=_hyperopt_worker, args=(hyper_args_list,))

        pb_proc.start()
        job.progress_bar = pb_proc

        hyper_proc.start()
        job.process = hyper_proc

        while hyper_proc.is_alive():
            hyper_proc.join(timeout=0.5)

        job.process = None
        cancelled = job.cancelled

        if cancelled or hyper_proc.exitcode not in (0, None):
            if pb_proc.is_alive():
                pb_proc.terminate()
                pb_proc.join(timeout=2)
            end_job(job)
            if cancelled:
                return render_hyperopt(warnings=['Hyperopt was cancelled.'])
            errors.append('Hyperopt failed — check server logs for details.')
            return render_hyperopt(warnings=warnings, errors=errors)

        job.progress.value = 100
        pb_proc.join()
        end_job(job)

        if backend == 'v2':
            # Still inside the run's TemporaryDirectory: chemprop 2 writes its
            # chosen settings under the save directory, which is about to be
            # removed, so the copy has to happen before leaving this block.
            produced = backends.hpopt_best_config(hyperopt_save_dir)
            if produced is None:
                errors.append(f'Hyperopt finished without producing a configuration — see '
                              f'{os.path.join(app.config["TEMP_FOLDER"], "hyperopt.log")}.')
                return render_hyperopt(warnings=warnings, errors=errors)
            shutil.copy(produced, config_save_path)

    # Load best hyperparams from saved config
    if backend == 'v2':
        best_config = backends.read_config_file(config_save_path)
    else:
        with open(config_save_path) as f:
            best_config = json.load(f)

    return render_hyperopt(
        completed=True,
        best_config=best_config,
        warnings=warnings,
        errors=errors,
    )


def parse_smiles_text(text: str) -> Tuple[List[str], List[Optional[str]]]:
    """Parse a block of text into (smiles, identifiers).

    Each line may be bare SMILES or SMILES followed by an identifier separated
    by a tab, comma, or whitespace.  Returns two parallel lists; entries with
    no identifier get None.
    """
    smiles_list: List[str] = []
    ids_list: List[Optional[str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if '\t' in line:
            parts = line.split('\t', 1)
        elif ',' in line:
            parts = line.split(',', 1)
        else:
            parts = line.split(None, 1)
        smiles_list.append(parts[0].strip())
        ids_list.append(parts[1].strip() if len(parts) > 1 else None)
    return smiles_list, ids_list


def render_predict(**kwargs):
    """Renders the predict page with specified kwargs"""
    checkpoint_upload_warnings, checkpoint_upload_errors = get_upload_warnings_errors('checkpoint')

    return render_template('predict.html',
                           checkpoints=db.get_ckpts(current_user_id()),
                           cuda=app.config['CUDA'],
                           gpus=app.config['GPUS'],
                           checkpoint_upload_warnings=checkpoint_upload_warnings,
                           checkpoint_upload_errors=checkpoint_upload_errors,
                           users=db.get_all_users(),
                           **kwargs)


def run_v2_prediction(job, smiles, model_paths, task_names, gpu, ensemble_unc,
                      use_conformal_reg, cal_path, conformal_alpha,
                      calibration_smiles=None):
    """Predicts with a chemprop 2.x checkpoint by running its CLI.

    Returns ``(result, failed)``, where ``result`` has the same shape the v1
    backend's ``predict_worker`` puts on its queue, so the response rendering is
    shared between the two backends. The running subprocess is published as
    the job so /cancel can stop it, exactly like the v1 path.
    """

    temp_folder = app.config['TEMP_FOLDER']
    env = backends.subprocess_env(gpu)
    accelerator = backends.accelerator_for(gpu)
    log_path = os.path.join(temp_folder, 'v2_predict.log')

    def _run(query_smiles, preds_filename, uncertainty_method=None,
             calibration_method=None):
        mask = _v2_valid_mask(query_smiles)
        if not any(mask):
            return [None] * len(mask), {}

        query_path = backends.write_smiles_csv(
            [s for s, ok in zip(query_smiles, mask) if ok],
            os.path.join(temp_folder, f'v2_query_{preds_filename}'))
        preds_path = os.path.join(temp_folder, preds_filename)

        cmd = backends.build_predict_cmd(
            app.config['CHEMPROP2_BIN'],
            test_path=query_path, preds_path=preds_path, model_paths=model_paths,
            uncertainty_method=uncertainty_method,
            cal_path=cal_path if calibration_method else None,
            calibration_method=calibration_method,
            conformal_alpha=conformal_alpha if calibration_method else None,
            accelerator=accelerator)

        proc = backends.run_cli(cmd, log_path, env)
        job.process = proc
        while proc.is_alive():
            proc.join(timeout=0.5)
        job.process = None

        if proc.exitcode != 0:
            raise backends.BackendError(
                f'chemprop 2 prediction failed (exit code {proc.exitcode}); '
                f'see {log_path}')

        preds, extras = backends.read_preds(preds_path, task_names)
        return (_v2_restore_invalid(preds, mask),
                {name: _v2_restore_invalid(values, mask)
                 for name, values in extras.items()})

    try:
        if use_conformal_reg:
            preds, extras = _run(smiles, app.config['PREDICTIONS_FILENAME'],
                                 uncertainty_method=backends.conformal_uncertainty_method(
                                     len(model_paths)),
                                 calibration_method='conformal-regression')
            unc = _v2_conformal_halfwidths(extras, task_names)
        elif ensemble_unc:
            preds, extras = _run(smiles, app.config['PREDICTIONS_FILENAME'],
                                 uncertainty_method='ensemble')
            unc = backends.column_group(extras, task_names, backends.UNCERTAINTY_SUFFIXES)
        else:
            preds, _ = _run(smiles, app.config['PREDICTIONS_FILENAME'])
            unc = None

        result = {'success': True, 'preds': preds, 'unc': unc}

        # Mondrian conformal for classification needs plain probabilities on the
        # calibration set as well; see mondrian_conformal_thresholds.
        if calibration_smiles:
            cal_preds, _ = _run(calibration_smiles, 'v2_calibration_preds.csv')
            result['cal_preds'] = cal_preds

        return result, False
    except backends.BackendError as e:
        job.process = None
        return {'success': False, 'error': str(e)}, True


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Renders the predict page and makes predictions if the method is POST."""
    if request.method == 'GET':
        return render_predict()

    # Get arguments
    ckpt_id = request.form['checkpointName']
    if owned_ckpt(ckpt_id) is None:
        return render_predict(errors=['That checkpoint is not available.'])

    identifiers: Optional[List[Optional[str]]] = None

    if request.form['textSmiles'] != '':
        raw_smiles, identifiers = parse_smiles_text(request.form['textSmiles'])
        smiles = raw_smiles
    elif request.form['drawSmiles'] != '':
        smiles = [request.form['drawSmiles']]
    else:
        # Upload data file with SMILES
        data = request.files['data']
        data_name = secure_filename(data.filename)
        data_path = os.path.join(app.config['TEMP_FOLDER'], data_name)
        data.save(data_path)

        # Check if header is smiles
        possible_smiles = get_header(data_path)[0]
        smiles = [possible_smiles] if Chem.MolFromSmiles(possible_smiles) is not None else []

        # Get remaining smiles
        smiles.extend(get_smiles(data_path))

    smiles = [[s] for s in smiles]

    models = db.get_models(ckpt_id)
    model_paths = [os.path.join(app.config['CHECKPOINT_FOLDER'], f'{model["id"]}.pt') for model in models]

    gpu = request.form.get('gpu')
    backend = ckpt_backend(ckpt_id)
    meta = load_ckpt_meta(ckpt_id)

    if backend == 'v2':
        # A chemprop 2.x checkpoint can only be opened by chemprop 2.x, so everything
        # this page needs about the model comes from the sidecar written at train time.
        if not meta or not meta.get('task_names'):
            return render_predict(errors=[
                'This chemprop 2 checkpoint is missing its metadata file, so it cannot '
                'be used for prediction. Retrain it to restore the metadata.'])
        task_names = meta['task_names']
        model_dataset_type = meta.get('dataset_type')
        train_args = None
    else:
        task_names = load_task_names(model_paths[0])
        train_args = load_args(model_paths[0])
        model_dataset_type = train_args.dataset_type

    num_tasks = len(task_names)

    # Conformal prediction (optional): needs a calibration set saved at train time
    # and a regression/classification model. Falls back to standard output if either
    # is missing, surfacing a warning to the user.
    conformal_enabled, conformal_alpha = parse_conformal_form(request.form)
    cal_path = conformal_calibration_path(ckpt_id)
    pred_warnings = []
    use_conformal = False
    if conformal_enabled:
        if model_dataset_type not in ('regression', 'classification'):
            pred_warnings.append('Conformal prediction is not available for this model type; showing standard predictions.')
        elif not os.path.exists(cal_path):
            pred_warnings.append('Conformal prediction is unavailable for this checkpoint — no calibration set was saved when it was trained. Showing standard predictions.')
        else:
            use_conformal = True

    # Regression conformal uses chemprop's conformal_regression directly. Classification
    # conformal uses class-conditional (Mondrian) calibration computed here from plain
    # probabilities, so it just needs predictions on the query and calibration sets.
    use_conformal_reg = use_conformal and model_dataset_type == 'regression'
    use_conformal_cls = use_conformal and model_dataset_type == 'classification'
    cal_smiles = cal_labels = None
    if use_conformal_cls:
        cal_smiles, cal_labels = load_calibration_data(cal_path, task_names)

    # Conformal and ensemble std-dev are mutually exclusive output modes; conformal
    # takes precedence. Query uncertainty is only returned for regression-conformal
    # (intervals) or the ensemble ± std.
    ensemble_unc = len(model_paths) > 1 and not use_conformal
    return_unc = ensemble_unc or use_conformal_reg
    arguments = [
        '--test_path', 'None',  # v1 backend only; the v2 backend builds its own command
        '--preds_path', os.path.join(app.config['TEMP_FOLDER'], app.config['PREDICTIONS_FILENAME']),
        '--checkpoint_paths', *model_paths
    ]
    if use_conformal_reg:
        arguments += [
            # Name the calibration file's SMILES column explicitly: when predicting from
            # an in-memory SMILES list there is no test CSV to auto-detect it from, so
            # smiles_columns would otherwise be [None] and the calibration load would fail.
            '--smiles_columns', 'smiles',
            '--calibration_path', cal_path,
            '--calibration_method', 'conformal_regression',
            '--conformal_alpha', str(conformal_alpha),
        ]
    elif ensemble_unc:
        arguments += ['--uncertainty_method', 'ensemble']

    if gpu is not None:
        if gpu == 'None':
            arguments.append('--no_cuda')
        else:
            arguments += ['--gpu', gpu]

    # Handle additional features (chemprop 1.x checkpoints only; the v2 backend
    # does not offer molecule-level feature generators through this app)
    if train_args is not None:
        if train_args.features_path is not None:
            # TODO: make it possible to specify the features generator if trained using features_path
            arguments += [
                '--features_generator', 'rdkit_2d_normalized',
                '--no_features_scaling'
            ]
        elif train_args.features_generator is not None:
            arguments += ['--features_generator', *train_args.features_generator]

            if not train_args.features_scaling:
                arguments.append('--no_features_scaling')

    job = start_job('predict')

    if backend == 'v2':
        result, failed = run_v2_prediction(
            job, smiles=smiles, model_paths=model_paths, task_names=task_names, gpu=gpu,
            ensemble_unc=ensemble_unc, use_conformal_reg=use_conformal_reg,
            cal_path=cal_path, conformal_alpha=conformal_alpha,
            calibration_smiles=cal_smiles)
    else:
        # Run predictions in a subprocess so they can be cancelled
        result_queue = _spawn.Queue()
        pred_proc = _spawn.Process(target=_predict_worker, args=(arguments, smiles, result_queue, return_unc, cal_smiles))
        pred_proc.start()
        job.process = pred_proc

        while pred_proc.is_alive():
            pred_proc.join(timeout=0.5)

        job.process = None
        failed = pred_proc.exitcode != 0
        result = None if failed else result_queue.get_nowait()

    cancelled = job.cancelled
    end_job(job)

    if cancelled:
        return render_predict(warnings=['Prediction was cancelled.'])
    if failed and result is None:
        return render_predict(errors=['Prediction failed — check server logs for details.'])

    if not result['success']:
        return render_predict(errors=[result['error']])
    preds = result['preds']
    raw_unc = result.get('unc')
    cal_preds = result.get('cal_preds')

    if all(p is None for p in preds):
        return render_predict(errors=['All SMILES are invalid'])

    # Convert per-task ensemble variance → std dev (rounded); None for invalid entries.
    # Values may be numpy.float32 which fails isinstance(v, float), so use float() conversion.
    def _var_to_std(row):
        if row is None:
            return None
        result = []
        for v in row:
            try:
                fv = float(v)
                result.append(round(math.sqrt(fv), 3) if fv >= 0 else None)
            except (TypeError, ValueError):
                result.append(None)
        return result

    # When conformal is active, raw_unc holds conformal output rather than ensemble
    # variance, so the ± std column is suppressed and replaced by intervals/sets.
    unc_std = None if use_conformal else (
        [_var_to_std(row) for row in raw_unc] if raw_unc is not None else None)

    conformal_result = None
    if use_conformal_reg and raw_unc is not None:
        # raw_unc[i][t] is the half-interval; interval is [pred - half, pred + half].
        intervals = []
        for i, pred in enumerate(preds):
            row = raw_unc[i] if raw_unc[i] is not None else []
            cell = []
            for ti in range(num_tasks):
                try:
                    mid = float(pred[ti])
                    half = float(row[ti])
                    cell.append((round(mid - half, 3), round(mid + half, 3)))
                except (TypeError, ValueError, IndexError):
                    cell.append(None)
            intervals.append(cell)
        conformal_result = {'mode': 'regression', 'alpha': conformal_alpha, 'intervals': intervals}
    elif use_conformal_cls and cal_preds is not None:
        # Mondrian: per-class thresholds from the calibration set's probabilities,
        # then a class-conditional prediction set for each query molecule.
        thr = mondrian_conformal_thresholds(cal_preds, cal_labels, conformal_alpha)
        categories = []
        for i in range(len(preds)):
            cell = []
            for ti in range(num_tasks):
                t = thr[ti] if ti < len(thr) else {'q_active': None, 'q_inactive': None}
                try:
                    p = float(preds[i][ti])
                except (TypeError, ValueError, IndexError):
                    cell.append(None)
                    continue
                active_in = t['q_active'] is not None and (1.0 - p) <= t['q_active']
                inactive_in = t['q_inactive'] is not None and p <= t['q_inactive']
                cell.append({'cat': mondrian_category(p, t),
                             'active_in': int(active_in), 'inactive_in': int(inactive_in)})
            categories.append(cell)
        conformal_result = {'mode': 'classification', 'method': 'mondrian',
                            'alpha': conformal_alpha, 'categories': categories}

    # Replace invalid smiles with message
    invalid_smiles_warning = 'Invalid SMILES String'
    preds = [pred if pred is not None else [invalid_smiles_warning] * num_tasks for pred in preds]

    # Compute atom attribution SVGs (best-effort; failures yield None). Attribution
    # reads chemprop 1.x model internals, so v2 checkpoints get no highlighting.
    flat_smiles = [s[0] for s in smiles[:10]]
    if backend == 'v2':
        attribution_svgs, _ = v2_attribution_svgs(ckpt_id, model_paths, flat_smiles, gpu=gpu)
    else:
        device = None if (gpu is None or gpu == 'None') else torch.device(f'cuda:{gpu}')
        attribution_svgs = compute_attributions(model_paths, flat_smiles, device=device)

    # Estimate the applicability domain of each prediction by comparing every
    # query molecule to the model's training set (best-effort; needs the saved
    # train/test predictions CSV that records the training SMILES).
    applicability = None
    ad_threshold = None
    try:
        train_preds_path = os.path.join(app.config['CHECKPOINT_FOLDER'], f'{ckpt_id}_train_test_preds.csv')
        train_smiles = load_training_smiles(train_preds_path)
        if train_smiles:
            ad = ApplicabilityDomain.from_training_smiles(train_smiles)
            if ad is not None:
                applicability = ad.score_all([s[0] for s in smiles])
                ad_threshold = ad.threshold
    except Exception:
        applicability = None

    # Write predictions CSV with rounded values (3 d.p.), plus std dev columns if available
    has_ids = identifiers is not None and any(i is not None for i in identifiers)
    preds_path = os.path.join(app.config['TEMP_FOLDER'], app.config['PREDICTIONS_FILENAME'])
    with open(preds_path, 'w', newline='') as f:
        writer = csv.writer(f)
        std_cols = [f'std_{t}' for t in task_names] if unc_std is not None else []
        conf_cols = []
        if conformal_result is not None:
            if conformal_result['mode'] == 'regression':
                for t in task_names:
                    conf_cols += [f'pi_low_{t}', f'pi_high_{t}']
            else:
                for t in task_names:
                    conf_cols += [f'conformal_{t}', f'active_in_set_{t}', f'inactive_in_set_{t}']
        ad_cols = ['ad_similarity', 'ad_in_domain', 'ad_threshold'] if applicability is not None else []
        header = (['id'] if has_ids else []) + ['smiles'] + task_names + std_cols + conf_cols + ad_cols
        writer.writerow(header)
        for idx, (smi_row, pred) in enumerate(zip(smiles, preds)):
            row = []
            if has_ids:
                row.append(identifiers[idx] if identifiers[idx] is not None else '')
            row.append(smi_row[0])
            row.extend([round(v, 3) if isinstance(v, float) else v for v in pred])
            if unc_std is not None:
                urow = unc_std[idx] if unc_std[idx] is not None else [None] * num_tasks
                row.extend([v if v is not None else '' for v in urow])
            if conformal_result is not None:
                if conformal_result['mode'] == 'regression':
                    for cell in conformal_result['intervals'][idx]:
                        row.extend(['', ''] if cell is None else [cell[0], cell[1]])
                else:
                    for cell in conformal_result['categories'][idx]:
                        row.extend(['', '', ''] if cell is None else [cell['cat'], cell['active_in'], cell['inactive_in']])
            if applicability is not None:
                ad = applicability[idx]
                if ad is not None:
                    row.extend([ad['similarity'], 'yes' if ad['in_domain'] else 'no', ad['threshold']])
                else:
                    row.extend(['', '', ''])
            writer.writerow(row)

    if None in preds:
        pred_warnings.append("List contains invalid SMILES strings")

    return render_predict(predicted=True,
                          smiles=smiles,
                          num_smiles=min(10, len(smiles)),
                          show_more=max(0, len(smiles)-10),
                          task_names=task_names,
                          num_tasks=len(task_names),
                          preds=preds,
                          unc_std=unc_std,
                          conformal=conformal_result,
                          identifiers=identifiers,
                          applicability=applicability,
                          ad_threshold=round(ad_threshold, 3) if ad_threshold is not None else None,
                          attribution_svgs=attribution_svgs,
                          warnings=pred_warnings or None,
                          errors=["No SMILES strings given"] if len(preds) == 0 else None)


@app.route('/get_attribution')
def get_attribution():
    """Returns an atom attribution SVG for a given SMILES and checkpoint."""
    smiles = request.args.get('smiles', '')
    ckpt_id = request.args.get('ckpt_id', '')

    if not smiles or not ckpt_id:
        return jsonify({'error': 'Missing smiles or ckpt_id'}), 400

    if owned_ckpt(ckpt_id) is None:
        return jsonify({'error': 'Checkpoint not found'}), 404

    models = db.get_models(ckpt_id)
    if not models:
        return jsonify({'error': 'No models found'}), 404

    model_paths = [os.path.join(app.config['CHECKPOINT_FOLDER'], f'{m["id"]}.pt') for m in models]

    if ckpt_backend(ckpt_id) == 'v2':
        if Chem.MolFromSmiles(smiles) is None:
            return jsonify({'error': 'Could not render this molecule.'}), 400
        svgs, attributed = v2_attribution_svgs(ckpt_id, model_paths, [smiles],
                                               gpu=request.args.get('gpu'))
        svg = svgs[0] if svgs else None
        if svg is None:
            return jsonify({'error': 'Could not render this molecule.'}), 400
        # A plain depiction comes back when the weights could not be computed; the
        # legend must then not claim the colours mean anything.
        return jsonify({'svg': svg, 'has_attribution': attributed[0]})

    try:
        train_args = load_args(model_paths[0])
        has_attribution = train_args.features_generator is None
        svgs = compute_attributions(model_paths, [smiles])
        svg = svgs[0] if svgs else None
        if svg is None:
            return jsonify({'error': 'Attribution not available'}), 400
        return jsonify({'svg': svg, 'has_attribution': has_attribution})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download_predictions')
def download_predictions():
    """Downloads predictions as a .csv file."""
    return send_from_directory(app.config['TEMP_FOLDER'], app.config['PREDICTIONS_FILENAME'], as_attachment=True, cache_timeout=-1)


@app.route('/download_train_test_predictions')
@check_not_demo
def download_train_test_predictions():
    """Downloads the combined train/test predictions CSV."""
    return send_from_directory(app.config['TEMP_FOLDER'], app.config['TRAIN_TEST_PREDS_FILENAME'],
                               as_attachment=True, download_name='train_test_predictions.csv', cache_timeout=-1)


@app.route('/download_hyperopt_config')
@check_not_demo
def download_hyperopt_config():
    """Downloads the best hyperparameter config from the last hyperopt run as a .json file,
    named ``<dataset_name>_hyperopt.json`` after the dataset that was optimized."""
    settings = LAST_HYPEROPT_SETTINGS.get(str(current_user_id() or ''), {})
    # chemprop 1.x writes JSON, chemprop 2 a config file for its own parser.
    ext = 'toml' if settings.get('backend') == 'v2' else 'json'
    filename = f'hyperopt_config.{ext}'

    download_name = filename
    ds_id = settings.get('dataName')
    if ds_id:
        row = db.query_db('SELECT dataset_name FROM dataset WHERE id = ?', (ds_id,), one=True)
        if row and row['dataset_name']:
            safe = secure_filename(row['dataset_name']) or 'dataset'
            download_name = f'{safe}_hyperopt.{ext}'
    return send_from_directory(app.config['TEMP_FOLDER'], filename, as_attachment=True,
                               download_name=download_name, cache_timeout=-1)


@app.route('/data')
@check_not_demo
def data():
    """Renders the data page."""
    data_upload_warnings, data_upload_errors = get_upload_warnings_errors('data')

    return render_template('data.html',
                           datasets=db.get_datasets(current_user_id()),
                           data_upload_warnings=data_upload_warnings,
                           data_upload_errors=data_upload_errors,
                           users=db.get_all_users())


@app.route('/data/upload/<string:return_page>', methods=['POST'])
@check_not_demo
def upload_data(return_page: str):
    """
    Uploads a data .csv file.

    :param return_page: The name of the page to render to after uploading the dataset.
    """
    warnings, errors = [], []

    current_user = current_user_id()

    if not current_user:
        # Use DEFAULT as current user if the client's cookie is not set.
        current_user = app.config['DEFAULT_USER_ID']

    dataset = request.files['dataset']

    upload_id_col = request.form.get('idColumn', '').strip() or None
    with NamedTemporaryFile() as temp_file:
        dataset.save(temp_file.name)
        try:
            _normalize_to_csv(temp_file.name)
        except (OSError, UnicodeDecodeError, csv.Error) as e:
            errors.append(f'Could not read uploaded file: {e}')
            warnings, errors = json.dumps(warnings), json.dumps(errors)
            return redirect(url_for(return_page, data_upload_warnings=warnings, data_upload_errors=errors))

        dataset_errors = validate_data(temp_file.name, ignore_columns=[upload_id_col] if upload_id_col else None)

        if len(dataset_errors) > 0:
            errors.extend(dataset_errors)
        else:
            dataset_name = request.form['datasetName']
            # dataset_class = load_args(ckpt).dataset_type  # TODO: SWITCH TO ACTUALLY FINDING THE CLASS

            dataset_id, new_dataset_name = db.insert_dataset(dataset_name, current_user, 'UNKNOWN')

            dataset_path = os.path.join(app.config['DATA_FOLDER'], f'{dataset_id}.csv')

            if dataset_name != new_dataset_name:
                warnings.append(name_already_exists_message('Data', dataset_name, new_dataset_name))

            shutil.copy(temp_file.name, dataset_path)

    warnings, errors = json.dumps(warnings), json.dumps(errors)

    return redirect(url_for(return_page, data_upload_warnings=warnings, data_upload_errors=errors))


@app.route('/data/download/<int:dataset>')
@check_not_demo
def download_data(dataset: int):
    """
    Downloads a dataset as a .csv file.

    :param dataset: The id of the dataset to download.
    """
    row = owned_dataset(dataset)
    if row is None:
        return 'Dataset not found', 404
    download_name = f'{row["dataset_name"]}.csv'
    return send_from_directory(app.config['DATA_FOLDER'], f'{dataset}.csv', as_attachment=True,
                               download_name=download_name, cache_timeout=-1)


@app.route('/data/delete/<int:dataset>')
@check_not_demo
def delete_data(dataset: int):
    """
    Deletes a dataset.

    :param dataset: The id of the dataset to delete.
    """
    if owned_dataset(dataset) is None:
        return 'Dataset not found', 404
    db.delete_dataset(dataset)
    os.remove(os.path.join(app.config['DATA_FOLDER'], f'{dataset}.csv'))
    return redirect(url_for('data'))


@app.route('/data/delete_all')
@check_not_demo
def delete_all_data():
    """Deletes all datasets belonging to the current user."""
    current_user = current_user_id() or app.config['DEFAULT_USER_ID']
    datasets = db.get_datasets(current_user)
    for dataset in datasets:
        db.delete_dataset(dataset['id'])
        path = os.path.join(app.config['DATA_FOLDER'], f'{dataset["id"]}.csv')
        if os.path.exists(path):
            os.remove(path)
    return redirect(url_for('data'))


@app.route('/data/rename/<int:dataset>', methods=['POST'])
@check_not_demo
def rename_data(dataset: int):
    if owned_dataset(dataset) is None:
        return jsonify(success=False, error='Dataset not found.'), 404
    new_name = request.form.get('name', '').strip()
    if not new_name:
        return jsonify(success=False, error='Name cannot be empty.')
    try:
        db.rename_dataset(dataset, new_name)
        return jsonify(success=True, name=new_name)
    except Exception:
        return jsonify(success=False, error='That name is already in use.')


@app.route('/checkpoint/<int:ckpt_id>/results')
@check_not_demo
def checkpoint_results(ckpt_id: int):
    """Returns the saved training results JSON for a checkpoint."""
    if owned_ckpt(ckpt_id) is None:
        return jsonify(error='No results saved for this checkpoint'), 404
    results_path = os.path.join(app.config['CHECKPOINT_FOLDER'], f'{ckpt_id}_results.json')
    if not os.path.exists(results_path):
        return jsonify(error='No results saved for this checkpoint'), 404
    with open(results_path) as _f:
        data = json.load(_f)
    preds_path = os.path.join(app.config['CHECKPOINT_FOLDER'], f'{ckpt_id}_train_test_preds.csv')
    data['has_preds_csv'] = os.path.exists(preds_path)
    warm_v2_attribution(ckpt_id)
    return jsonify(data)


@app.route('/checkpoint/<int:ckpt_id>/download_predictions')
@check_not_demo
def download_checkpoint_predictions(ckpt_id: int):
    """Downloads the permanent train/test predictions CSV for a checkpoint."""
    if owned_ckpt(ckpt_id) is None:
        return 'Predictions not available', 404
    preds_path = os.path.join(app.config['CHECKPOINT_FOLDER'], f'{ckpt_id}_train_test_preds.csv')
    if not os.path.exists(preds_path):
        return 'Predictions not available', 404
    return send_file(preds_path, as_attachment=True, download_name='train_test_predictions.csv')


@app.route('/checkpoints')
@check_not_demo
def checkpoints():
    """Renders the checkpoints page."""
    checkpoint_upload_warnings, checkpoint_upload_errors = get_upload_warnings_errors('checkpoint')
    all_ckpts = db.get_ckpts(current_user_id())
    ckpts_with_results = {
        ckpt['id'] for ckpt in all_ckpts
        if os.path.exists(os.path.join(app.config['CHECKPOINT_FOLDER'], f'{ckpt["id"]}_results.json'))
    }

    return render_template('checkpoints.html',
                           checkpoints=all_ckpts,
                           ckpts_with_results=ckpts_with_results,
                           checkpoint_upload_warnings=checkpoint_upload_warnings,
                           checkpoint_upload_errors=checkpoint_upload_errors,
                           users=db.get_all_users())


@app.route('/checkpoints/upload/<string:return_page>', methods=['POST'])
@check_not_demo
def upload_checkpoint(return_page: str):
    """
    Uploads a checkpoint .pt file.

    :param return_page: The name of the page to render after uploading the checkpoint file.
    """
    warnings, errors = [], []

    current_user = current_user_id()

    if not current_user:
        # Use DEFAULT as current user if the client's cookie is not set.
        current_user = app.config['DEFAULT_USER_ID']

    ckpt = request.files['checkpoint']

    ckpt_name = request.form['checkpointName']
    ckpt_ext = os.path.splitext(ckpt.filename)[1]

    # Collect paths to all uploaded checkpoints (and unzip if necessary)
    temp_dir = TemporaryDirectory()
    ckpt_paths = []

    if ckpt_ext.endswith('.pt'):
        ckpt_path = os.path.join(temp_dir.name, MODEL_FILE_NAME)
        ckpt.save(ckpt_path)
        ckpt_paths = [ckpt_path]

    elif ckpt_ext.endswith('.zip'):
        ckpt_dir = os.path.join(temp_dir.name, 'models')
        zip_path = os.path.join(temp_dir.name, 'models.zip')
        ckpt.save(zip_path)

        with zipfile.ZipFile(zip_path, mode='r') as z:
            for member in z.namelist():
                member_path = os.path.realpath(os.path.join(ckpt_dir, member))
                if not member_path.startswith(os.path.realpath(ckpt_dir)):
                    errors.append('Invalid zip file: contains path traversal entry.')
                    break
            else:
                z.extractall(ckpt_dir)

        for root, _, fnames in os.walk(ckpt_dir):
            ckpt_paths += [os.path.join(root, fname) for fname in fnames if fname.endswith('.pt')]

    else:
        errors.append(f'Uploaded checkpoint(s) file must be either .pt or .zip but got {ckpt_ext}')

    # Insert checkpoints into database
    if len(ckpt_paths) > 0:
        try:
            ckpt_args = load_args(ckpt_paths[0])
        except Exception:
            # A chemprop 2 checkpoint carries no chemprop 1.x arguments; it is
            # identified and described by the v2 environment instead.
            ckpt_args = None

        backend = 'v1' if ckpt_args is not None else 'v2'
        info = task_names = None

        if backend == 'v2':
            if not app.config['CHEMPROP2_AVAILABLE']:
                temp_dir.cleanup()
                errors.append('This is not a chemprop 1 checkpoint, and no chemprop 2 '
                              'environment is available to read it.')
                return redirect(url_for(return_page,
                                        checkpoint_upload_warnings=json.dumps(warnings),
                                        checkpoint_upload_errors=json.dumps(errors)))

            info = backends.checkpoint_info(
                app.config['CHEMPROP2_PYTHON'], v2_checkpoint_info_script(),
                ckpt_paths[0], backends.subprocess_env(None))

            try:
                task_names = validate_uploaded_v2(info, request.form.get('taskNames', ''))
            except ValueError as e:
                temp_dir.cleanup()
                errors.append(str(e))
                return redirect(url_for(return_page,
                                        checkpoint_upload_warnings=json.dumps(warnings),
                                        checkpoint_upload_errors=json.dumps(errors)))

        if backend == 'v1':
            model_class, epochs, training_size = (ckpt_args.dataset_type, ckpt_args.epochs,
                                                  ckpt_args.train_data_size)
        else:
            # An uploaded model does not say how long it trained or on how much;
            # zero reads as "unknown" on the Checkpoints page.
            model_class, epochs, training_size = info['task_type'], 0, 0

        ckpt_id, new_ckpt_name = db.insert_ckpt(ckpt_name,
                                                current_user,
                                                model_class,
                                                epochs,
                                                len(ckpt_paths),
                                                training_size,
                                                backend=backend)

        for ckpt_path in ckpt_paths:
            model_id = db.insert_model(ckpt_id)
            model_path = os.path.join(app.config['CHECKPOINT_FOLDER'], f'{model_id}.pt')

            if ckpt_name != new_ckpt_name:
                warnings.append(name_already_exists_message('Checkpoint', ckpt_name, new_ckpt_name))

            shutil.copy(ckpt_path, model_path)

        if backend == 'v2':
            # Predicting from a v2 checkpoint reads this rather than the model file.
            write_ckpt_meta(ckpt_id, backend='v2', foundation=None,
                            task_names=task_names, dataset_type=info['task_type'],
                            features_generator=None, smiles_column='smiles')
            # A foundation model has hundreds of targets; naming them all would
            # bury the part of this message that matters.
            described = (', '.join(task_names) if len(task_names) <= 6
                         else f'{len(task_names)} targets')
            warnings.append(
                f'Registered as a chemprop 2 model predicting {described}. Conformal '
                f'prediction and the applicability domain need a training set, so they '
                f'are unavailable for uploads.')

    temp_dir.cleanup()

    warnings, errors = json.dumps(warnings), json.dumps(errors)

    return redirect(url_for(return_page, checkpoint_upload_warnings=warnings, checkpoint_upload_errors=errors))


@app.route('/checkpoints/download/<int:checkpoint>')
@check_not_demo
def download_checkpoint(checkpoint: int):
    """
    Downloads a zip of model .pt files.

    :param checkpoint: The name of the checkpoint to download.
    """
    ckpt = owned_ckpt(checkpoint)
    if ckpt is None:
        return 'Checkpoint not found', 404
    models = db.get_models(checkpoint)

    model_data = io.BytesIO()

    with zipfile.ZipFile(model_data, mode='w') as z:
        for model in models:
            model_path = os.path.join(app.config['CHECKPOINT_FOLDER'], f'{model["id"]}.pt')
            z.write(model_path, os.path.basename(model_path))

    model_data.seek(0)

    return send_file(
        model_data,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'{ckpt["ckpt_name"]}.zip',
        cache_timeout=-1
    )


@app.route('/checkpoints/delete/<int:checkpoint>')
@check_not_demo
def delete_checkpoint(checkpoint: int):
    """
    Deletes a checkpoint file.

    :param checkpoint: The id of the checkpoint to delete.
    """
    if owned_ckpt(checkpoint) is None:
        return 'Checkpoint not found', 404
    for suffix in ('_results.json', '_train_test_preds.csv', '_calibration.csv', '_meta.json'):
        sidecar = os.path.join(app.config['CHECKPOINT_FOLDER'], f'{checkpoint}{suffix}')
        if os.path.exists(sidecar):
            os.remove(sidecar)
    db.delete_ckpt(checkpoint)
    return redirect(url_for('checkpoints'))


@app.route('/checkpoints/delete_all')
@check_not_demo
def delete_all_checkpoints():
    """Deletes all checkpoints belonging to the current user."""
    current_user = current_user_id() or app.config['DEFAULT_USER_ID']
    ckpts = db.get_ckpts(current_user)
    for ckpt in ckpts:
        db.delete_ckpt(ckpt['id'])
    return redirect(url_for('checkpoints'))


@app.route('/checkpoints/rename/<int:checkpoint>', methods=['POST'])
@check_not_demo
def rename_checkpoint(checkpoint: int):
    if owned_ckpt(checkpoint) is None:
        return jsonify(success=False, error='Checkpoint not found.'), 404
    new_name = request.form.get('name', '').strip()
    if not new_name:
        return jsonify(success=False, error='Name cannot be empty.')
    try:
        db.rename_ckpt(checkpoint, new_name)
        return jsonify(success=True, name=new_name)
    except Exception:
        return jsonify(success=False, error='That name is already in use.')
