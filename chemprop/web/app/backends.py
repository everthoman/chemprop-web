"""Driver for the chemprop 2.x backend.

Foundation models such as CheMeleon are chemprop 2.x artifacts: their weights are
a ``chemprop.nn.BondMessagePassing`` state dict paired with the v2 featurizer, and
neither can be loaded by the chemprop 1.x code this web app is built on. Rather
than importing two incompatible versions of chemprop into one interpreter, the v2
backend runs the ``chemprop`` CLI installed in a separate conda environment as a
subprocess. This process never imports chemprop 2.x.

Everything here is plain command building and file parsing (no Flask, no chemprop
imports) so it can be exercised on its own. The parsing helpers deliberately
return the same list-of-lists shapes that chemprop 1.x's ``make_predictions``
returns, so the views and templates consuming them need no v2-specific handling.
"""

import csv
import glob
import os
import signal
import subprocess
from typing import Dict, List, Optional, Sequence, Tuple

# Column-name suffixes chemprop 2.x appends to a target column when predicting with
# uncertainty or calibration. Several spellings are accepted because they vary by
# uncertainty method; the first group that matches every task wins.
UNCERTAINTY_SUFFIXES = ('_unc', '_uncertainty', '_uncal_var', '_var', '_mve_uncal_var',
                        '_ensemble_uncal_var')
INTERVAL_LOWER_SUFFIXES = ('_conformal_lower', '_interval_lower', '_lower')
INTERVAL_UPPER_SUFFIXES = ('_conformal_upper', '_interval_upper', '_upper')

# Name of the SMILES column written into the CSVs handed to the v2 CLI.
SMILES_COLUMN = 'smiles'


class BackendError(RuntimeError):
    """Raised when a chemprop 2.x subprocess fails or its output can't be read."""


# --- command building -----------------------------------------------------

def base_cmd(conda_exe: str, env_name: str) -> List[str]:
    """The prefix that runs the chemprop CLI inside the v2 conda environment."""
    return [conda_exe, 'run', '-n', env_name, '--no-capture-output', 'chemprop']


def build_train_cmd(conda_exe: str,
                    env_name: str,
                    data_path: str,
                    output_dir: str,
                    task_type: str,
                    task_names: Sequence[str],
                    smiles_column: str,
                    epochs: int,
                    ensemble_size: int,
                    split_type: str,
                    seed: int,
                    foundation: Optional[str] = None,
                    patience: Optional[int] = None,
                    accelerator: str = 'cpu') -> List[str]:
    """Builds a ``chemprop train`` command line.

    Target columns are passed explicitly rather than letting chemprop infer them,
    so an identifier column present in the CSV is excluded the same way the v1
    backend excludes it via ``ignore_columns``.
    """
    cmd = base_cmd(conda_exe, env_name) + [
        'train',
        '--data-path', data_path,
        '--output-dir', output_dir,
        '--task-type', task_type,
        '--smiles-columns', smiles_column,
        '--target-columns', *task_names,
        '--epochs', str(epochs),
        '--ensemble-size', str(ensemble_size),
        '--split', split_type.upper(),
        '--data-seed', str(seed),
        '--pytorch-seed', str(seed),
        # Data loading happens inside this subprocess; extra worker processes buy
        # nothing for the dataset sizes this app handles and complicate teardown.
        '--num-workers', '0',
        # Writes the full train/val/test splits (SMILES *and* targets) to disk. The
        # val split becomes the conformal calibration set and the train/test splits
        # drive the post-training plots, mirroring what the v1 backend computes.
        '--save-data-splits',
        '--accelerator', accelerator,
    ]

    if foundation:
        cmd += ['--from-foundation', foundation]

    if patience:
        cmd += ['--patience', str(patience)]

    return cmd


def build_predict_cmd(conda_exe: str,
                      env_name: str,
                      test_path: str,
                      preds_path: str,
                      model_paths: Sequence[str],
                      uncertainty_method: Optional[str] = None,
                      cal_path: Optional[str] = None,
                      calibration_method: Optional[str] = None,
                      conformal_alpha: Optional[float] = None,
                      accelerator: str = 'cpu') -> List[str]:
    """Builds a ``chemprop predict`` command line."""
    cmd = base_cmd(conda_exe, env_name) + [
        'predict',
        '--test-path', test_path,
        '--preds-path', preds_path,
        '--model-paths', *model_paths,
        '--smiles-columns', SMILES_COLUMN,
        '--num-workers', '0',
        '--accelerator', accelerator,
    ]

    if uncertainty_method:
        cmd += ['--uncertainty-method', uncertainty_method]

    if cal_path and calibration_method:
        cmd += ['--cal-path', cal_path, '--calibration-method', calibration_method]
        if conformal_alpha is not None:
            cmd += ['--conformal-alpha', str(conformal_alpha)]

    return cmd


def subprocess_env(gpu: Optional[str]) -> Dict[str, str]:
    """Environment for a v2 subprocess.

    chemprop 2.x selects devices through Lightning's ``--accelerator``/``--devices``
    and has no equivalent of v1's ``--gpu <index>``, so a specific GPU is chosen by
    masking the others.
    """
    env = dict(os.environ)

    # This app runs from a source checkout of chemprop 1.x, which is on the parent's
    # PYTHONPATH. Inheriting it would put chemprop 1.x ahead of the v2 environment's
    # own install and the CLI would import the wrong chemprop entirely.
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)

    if gpu is None or gpu == 'None':
        env['CUDA_VISIBLE_DEVICES'] = ''
    else:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    return env


def conformal_uncertainty_method(n_models: int) -> str:
    """The uncertainty a conformal calibration can be layered on.

    chemprop 2.x conformalises an existing uncertainty estimate rather than
    producing one, and refuses to calibrate when no uncertainty method is given.
    An ensemble supplies it when there is more than one model; a single model
    falls back to Monte-Carlo dropout.
    """
    return 'ensemble' if n_models > 1 else 'dropout'


def accelerator_for(gpu: Optional[str]) -> str:
    return 'cpu' if gpu is None or gpu == 'None' else 'gpu'


# --- running --------------------------------------------------------------

class Subprocess:
    """A ``Popen`` wearing the parts of the ``multiprocessing.Process`` interface
    that the views and the /cancel endpoint use.

    The v1 backend runs training in an ``mp.Process``; presenting the same surface
    here means cancellation, exit-code checks and the "was it killed?" logic work
    for both backends without branching. Signals go to the whole process group
    because ``conda run`` spawns the real chemprop process as a child, and killing
    only the parent would leave training running.
    """

    def __init__(self, popen: subprocess.Popen, log_file):
        self._popen = popen
        self._log_file = log_file

    def _signal_group(self, sig) -> None:
        try:
            os.killpg(os.getpgid(self._popen.pid), sig)
        except (ProcessLookupError, PermissionError):
            self._popen.send_signal(sig)

    def start(self) -> None:
        """No-op: ``Popen`` is already running. Present so callers can treat this
        like the ``mp.Process`` the v1 backend uses."""

    def is_alive(self) -> bool:
        return self._popen.poll() is None

    def terminate(self) -> None:
        self._signal_group(signal.SIGTERM)

    def kill(self) -> None:
        self._signal_group(signal.SIGKILL)

    def join(self, timeout: Optional[float] = None) -> None:
        try:
            self._popen.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass
        if not self.is_alive() and not self._log_file.closed:
            self._log_file.close()

    @property
    def exitcode(self) -> Optional[int]:
        """Matches ``mp.Process.exitcode``: None while running, negative if signalled."""
        return self._popen.returncode


def run_cli(cmd: Sequence[str], log_path: str, env: Dict[str, str]) -> Subprocess:
    """Starts a chemprop 2.x subprocess with its output tee'd to ``log_path``."""
    work_dir = os.path.dirname(log_path) or '.'
    os.makedirs(work_dir, exist_ok=True)
    log_file = open(log_path, 'w')
    # Run outside the chemprop 1.x checkout for the same reason PYTHONPATH is
    # dropped: nothing of this app's source should be importable by the v2 CLI.
    popen = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                             env=env, cwd=work_dir, start_new_session=True)
    return Subprocess(popen, log_file)


# --- output inspection ----------------------------------------------------

def collect_models(output_dir: str) -> List[str]:
    """Paths to the trained model files of a v2 run, one per ensemble member.

    A v2 output directory also holds Lightning ``.ckpt`` files and other artifacts,
    so only the ``model_*/best.pt`` files are picked up.
    """
    return sorted(glob.glob(os.path.join(output_dir, 'model_*', 'best.pt')))


def find_split_file(output_dir: str, split: str) -> Optional[str]:
    """Locates a saved split CSV (``train``/``val``/``test``) from a v2 run.

    The exact name at the top of the run directory is what ``--save-data-splits``
    writes. The recursive fallback deliberately skips prediction files, since a run
    directory also holds ``model_*/test_predictions.csv``, whose "targets" are the
    model's own predictions.
    """
    exact = os.path.join(output_dir, f'{split}.csv')
    if os.path.exists(exact):
        return exact

    matches = [path for path in
               sorted(glob.glob(os.path.join(output_dir, '**', f'{split}_*.csv'), recursive=True))
               if 'predictions' not in os.path.basename(path)]
    return matches[0] if matches else None


def epoch_progress(output_dir: str, total_epochs: int) -> float:
    """Percentage of training completed, from the Lightning logs of a v2 run.

    Counts the distinct epochs recorded across every ensemble member's metrics
    file. Falls back to 0 while the first epoch is still running (nothing is
    written until an epoch completes).
    """
    if total_epochs <= 0:
        return 0.0

    epochs_done = 0
    for metrics_csv in glob.glob(os.path.join(output_dir, 'model_*', '**', 'metrics.csv'),
                                 recursive=True):
        try:
            with open(metrics_csv) as f:
                seen = {row.get('epoch') for row in csv.DictReader(f)}
            epochs_done += len({e for e in seen if e not in (None, '')})
        except OSError:
            continue

    return min(epochs_done * 100.0 / total_epochs, 100.0)


def val_curves(output_dir: str) -> Dict:
    """Per-model validation scores by epoch, in the shape the Train page charts.

    Mirrors ``_parse_val_curves``'s output for the v1 backend, but reads chemprop
    2.x's Lightning ``metrics.csv`` instead of the v1 text log.
    """
    models: List[List[float]] = []
    metric_name = None

    for model_dir in sorted(glob.glob(os.path.join(output_dir, 'model_*'))):
        metrics_files = sorted(glob.glob(os.path.join(model_dir, '**', 'metrics.csv'),
                                         recursive=True))
        if not metrics_files:
            continue
        try:
            with open(metrics_files[0]) as f:
                rows = list(csv.DictReader(f))
        except OSError:
            continue
        if not rows:
            continue

        if metric_name is None:
            candidates = [c for c in rows[0]
                          if c.startswith(('val_', 'val/')) and 'loss' not in c]
            candidates = candidates or [c for c in rows[0] if c.startswith(('val_', 'val/'))]
            if not candidates:
                continue
            metric_name = candidates[0]

        if metric_name not in rows[0]:
            continue

        scores, seen = [], set()
        for row in rows:
            epoch, value = row.get('epoch'), _to_float(row.get(metric_name))
            if value is None or epoch in seen or epoch in (None, ''):
                continue
            seen.add(epoch)
            scores.append(value)
        if scores:
            models.append(scores)

    pretty = (metric_name or '').replace('val_', '').replace('val/', '')
    return {'metric': pretty, 'models': models}


def _to_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_preds(preds_path: str,
               task_names: Sequence[str]) -> Tuple[List[Optional[List[float]]], Dict[str, List[Optional[float]]]]:
    """Reads a v2 predictions CSV.

    :return: ``(preds, extras)`` where ``preds[i][t]`` is the prediction for row
             ``i`` and task ``t`` (matching chemprop 1.x's ``make_predictions``
             output), and ``extras`` maps every other column name to its values.
    """
    if not os.path.exists(preds_path):
        raise BackendError(f'chemprop 2 produced no predictions file at {preds_path}')

    with open(preds_path) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return [], {}

    missing = [t for t in task_names if t not in rows[0]]
    if missing:
        raise BackendError(
            f'Predictions file is missing column(s) {", ".join(missing)}; '
            f'got {", ".join(rows[0].keys())}')

    preds: List[Optional[List[float]]] = []
    for row in rows:
        values = [_to_float(row[t]) for t in task_names]
        preds.append(None if all(v is None for v in values) else values)

    extra_names = [c for c in rows[0]
                   if c not in task_names and c not in (SMILES_COLUMN, '')]
    extras = {name: [_to_float(row.get(name)) for row in rows] for name in extra_names}

    return preds, extras


def column_group(extras: Dict[str, List[Optional[float]]],
                 task_names: Sequence[str],
                 suffixes: Sequence[str]) -> Optional[List[Optional[List[float]]]]:
    """Assembles a per-task column group (e.g. uncertainties) out of ``extras``.

    Tries each suffix in turn and returns the first one that is present for every
    task, reshaped per row like ``read_preds``' predictions. Returns None when no
    suffix matches, which is how "chemprop didn't emit this" is reported.
    """
    for suffix in suffixes:
        columns = [f'{task}{suffix}' for task in task_names]
        if all(c in extras for c in columns):
            n_rows = len(extras[columns[0]])
            grouped: List[Optional[List[float]]] = []
            for i in range(n_rows):
                values = [extras[c][i] for c in columns]
                grouped.append(None if all(v is None for v in values) else values)
            return grouped
    return None


# --- CSV helpers ----------------------------------------------------------

def write_smiles_csv(smiles: Sequence, path: str) -> str:
    """Writes SMILES to a one-column CSV for the v2 CLI to read.

    Accepts either plain strings or the single-element lists the predict view uses.
    """
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([SMILES_COLUMN])
        for entry in smiles:
            writer.writerow([entry[0] if isinstance(entry, (list, tuple)) else entry])
    return path


def write_calibration_csv(split_path: str,
                          out_path: str,
                          smiles_column: str,
                          task_names: Sequence[str]) -> Optional[str]:
    """Converts a saved v2 validation split into this app's calibration format.

    The predict view expects a CSV with a ``smiles`` column plus one column per
    task (see ``load_calibration_data``), which is what the v1 backend writes.
    Returns the written path, or None when the split has no usable rows.
    """
    if not split_path or not os.path.exists(split_path):
        return None

    with open(split_path) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return None

    source_smiles = smiles_column if smiles_column in rows[0] else None
    if source_smiles is None:
        # Fall back to the first column that looks like SMILES.
        for candidate in (SMILES_COLUMN, 'SMILES', 'Smiles'):
            if candidate in rows[0]:
                source_smiles = candidate
                break
    if source_smiles is None:
        return None

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([SMILES_COLUMN, *task_names])
        for row in rows:
            writer.writerow([row.get(source_smiles, ''), *[row.get(t, '') for t in task_names]])

    return out_path
