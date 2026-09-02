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
import json
import os
import re
import selectors
import signal
import subprocess
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

# Column-name suffixes chemprop 2.x appends to a target column when predicting with
# uncertainty or calibration. Several spellings are accepted because they vary by
# uncertainty method; the first group that matches every task wins.
UNCERTAINTY_SUFFIXES = ('_unc', '_uncertainty', '_uncal_var', '_var', '_mve_uncal_var',
                        '_ensemble_uncal_var')
INTERVAL_LOWER_SUFFIXES = ('_conformal_lower', '_interval_lower', '_lower')
INTERVAL_UPPER_SUFFIXES = ('_conformal_upper', '_interval_upper', '_upper')

# The Train page offers chemprop 1.x's names for its extra molecule features;
# chemprop 2 spells them differently and keeps v1-compatible variants of the RDKit
# ones, which are the right choice here so the two backends compute the same thing.
MOLECULE_FEATURIZERS = {
    'rdkit_2d_normalized': 'v1_rdkit_2d_normalized',
    'rdkit_2d': 'v1_rdkit_2d',
    'morgan': 'morgan_binary',
    'morgan_count': 'morgan_count',
}

# Name of the SMILES column written into the CSVs handed to the v2 CLI.
SMILES_COLUMN = 'smiles'


class BackendError(RuntimeError):
    """Raised when a chemprop 2.x subprocess fails or its output can't be read."""


# --- command building -----------------------------------------------------

def base_cmd(chemprop_bin: str) -> List[str]:
    """The v2 environment's chemprop entry point.

    Invoked directly rather than through ``conda run``: the wrapper adds processes
    that end up outside the group this app signals, so a cancelled run would leave
    training alive. The console script's shebang selects the environment's Python,
    so no activation is needed.
    """
    return [chemprop_bin]


def build_train_cmd(chemprop_bin: str,
                    data_path: str,
                    output_dir: str,
                    task_type: str,
                    task_names: Sequence[str],
                    smiles_column: str,
                    epochs: int,
                    ensemble_size: int,
                    split_type: str,
                    seed: int,
                    split_sizes: Optional[Sequence[float]] = None,
                    splits_column: Optional[str] = None,
                    foundation: Optional[str] = None,
                    patience: Optional[int] = None,
                    min_delta: float = 0.0,
                    batch_size: Optional[int] = None,
                    config_path: Optional[str] = None,
                    molecule_featurizer: Optional[str] = None,
                    tracking_metric: Optional[str] = None,
                    accelerator: str = 'cpu') -> List[str]:
    """Builds a ``chemprop train`` command line.

    Target columns are passed explicitly rather than letting chemprop infer them,
    so an identifier column present in the CSV is excluded the same way the v1
    backend excludes it via ``ignore_columns``.
    """
    # The config comes first so that the explicit arguments below override the
    # settings a hyperopt run chose, notably the dataset and epoch count.
    cmd = base_cmd(chemprop_bin) + ['train']
    if config_path:
        cmd += ['--config-path', config_path]

    cmd += [
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

    if splits_column:
        # A column naming each row's split; the sizes are then whatever it says.
        cmd += ['--splits-column', splits_column]
    elif split_sizes:
        cmd += ['--split-sizes', *[str(part) for part in split_sizes]]

    if batch_size:
        cmd += ['--batch-size', str(batch_size)]

    if molecule_featurizer:
        cmd += ['--molecule-featurizers', molecule_featurizer]

    if tracking_metric:
        cmd += ['--tracking-metric', tracking_metric]

    if foundation:
        cmd += ['--from-foundation', foundation]

    if patience:
        cmd += ['--patience', str(patience)]
        if min_delta:
            cmd += ['--min-delta', str(min_delta)]

    return cmd


def build_predict_cmd(chemprop_bin: str,
                      test_path: str,
                      preds_path: str,
                      model_paths: Sequence[str],
                      uncertainty_method: Optional[str] = None,
                      cal_path: Optional[str] = None,
                      calibration_method: Optional[str] = None,
                      conformal_alpha: Optional[float] = None,
                      molecule_featurizer: Optional[str] = None,
                      accelerator: str = 'cpu') -> List[str]:
    """Builds a ``chemprop predict`` command line."""
    cmd = base_cmd(chemprop_bin) + [
        'predict',
        '--test-path', test_path,
        '--preds-path', preds_path,
        '--model-paths', *model_paths,
        '--smiles-columns', SMILES_COLUMN,
        '--num-workers', '0',
        '--accelerator', accelerator,
    ]

    if molecule_featurizer:
        cmd += ['--molecule-featurizers', molecule_featurizer]

    if uncertainty_method:
        cmd += ['--uncertainty-method', uncertainty_method]

    if cal_path and calibration_method:
        cmd += ['--cal-path', cal_path, '--calibration-method', calibration_method]
        if conformal_alpha is not None:
            cmd += ['--conformal-alpha', str(conformal_alpha)]

    return cmd


def build_hpopt_cmd(chemprop_bin: str,
                    data_path: str,
                    save_dir: str,
                    task_type: str,
                    task_names: Sequence[str],
                    smiles_column: str,
                    epochs: int,
                    num_trials: int,
                    search_keywords: Sequence[str],
                    split_sizes: Optional[Sequence[float]] = None,
                    search_algorithm: str = 'random',
                    foundation: Optional[str] = None,
                    batch_size: Optional[int] = None,
                    molecule_featurizer: Optional[str] = None,
                    tracking_metric: Optional[str] = None,
                    accelerator: str = 'cpu') -> List[str]:
    """Builds a ``chemprop hpopt`` command line."""
    cmd = base_cmd(chemprop_bin) + [
        'hpopt',
        '--data-path', data_path,
        '--hpopt-save-dir', save_dir,
        '--task-type', task_type,
        '--smiles-columns', smiles_column,
        '--target-columns', *task_names,
        '--epochs', str(epochs),
        '--raytune-num-samples', str(num_trials),
        '--raytune-search-algorithm', search_algorithm,
        '--search-parameter-keywords', *search_keywords,
        '--num-workers', '0',
        '--accelerator', accelerator,
    ]

    if accelerator == 'gpu':
        cmd.append('--raytune-use-gpu')
    if split_sizes:
        cmd += ['--split-sizes', *[str(part) for part in split_sizes]]
    if batch_size:
        cmd += ['--batch-size', str(batch_size)]
    if molecule_featurizer:
        cmd += ['--molecule-featurizers', molecule_featurizer]
    if tracking_metric:
        cmd += ['--tracking-metric', tracking_metric]
    if foundation:
        cmd += ['--from-foundation', foundation]

    return cmd


def hpopt_progress(log_path: str, num_trials: int) -> float:
    """Percentage of hyperopt trials finished, read from the run's output.

    Ray Tune prints a status table naming how many trials have terminated, which
    is the only progress signal written while the search is running: the results
    CSV is not saved until the end.
    """
    if num_trials <= 0 or not os.path.exists(log_path):
        return 0.0

    # Only the latest status matters, and Ray's output grows steadily, so read the
    # end of the file rather than all of it once a second.
    try:
        with open(log_path, 'rb') as f:
            f.seek(max(0, os.path.getsize(log_path) - 65536))
            text = f.read().decode('utf-8', errors='replace')
    except OSError:
        return 0.0

    done = 0
    for line in text.splitlines():
        if 'Trial status:' in line:
            match = re.search(r'(\d+) TERMINATED', line)
            if match:
                done = int(match.group(1))

    # Held below 100 so the caller decides when the run is actually finished.
    return min(done * 100.0 / num_trials, 99.0)


def hpopt_best_config(save_dir: str) -> Optional[str]:
    """Path to the configuration chemprop 2 chose, if the search produced one."""
    path = os.path.join(save_dir, 'best_config.toml')
    return path if os.path.exists(path) else None


def read_config_file(path: str) -> Dict[str, str]:
    """Reads a chemprop 2 config file into ``{setting: value}`` for display.

    Despite the .toml name these files are written for chemprop's own argument
    parser and their values are unquoted, so a TOML parser rejects them. Only the
    display needs the values; training reads the file itself.
    """
    settings = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, _, value = line.partition('=')
                settings[key.strip()] = value.strip().strip('[]')
    except OSError:
        return {}
    return settings


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

    # User site-packages precede an environment's own on sys.path, so a stray
    # ~/.local install of one of chemprop 2's dependencies would be picked up
    # instead of the environment's. Keep the environment self-contained.
    env['PYTHONNOUSERSITE'] = '1'

    # Output goes to a file, where Python would buffer it in 8 KB blocks: the
    # trainer's progress lines then reach the log minutes late, and the progress
    # bar reads them, so a running job looks like it has not started.
    env['PYTHONUNBUFFERED'] = '1'

    # Same device numbering as the dropdown was built from.
    env.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')

    if gpu is None or gpu == 'None':
        env['CUDA_VISIBLE_DEVICES'] = ''
    else:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    return env


# What early stopping and checkpoint selection should follow. chemprop 2 tracks
# the training loss by default, but for classification that is cross-entropy,
# which flattens while ranking is still improving - so a run stops with its AUC
# still climbing, and the checkpoint kept is the best-loss epoch rather than the
# best-AUC one. Regression's loss is the error being reported, so it needs no
# override.
TRACKING_METRICS = {'classification': 'roc'}


def tracking_metric_for(task_type: str) -> Optional[str]:
    """The validation metric a run should be judged on, or None for the default."""
    return TRACKING_METRICS.get(task_type)


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
    try:
        # Run outside the chemprop 1.x checkout for the same reason PYTHONPATH is
        # dropped: nothing of this app's source should be importable by the v2 CLI.
        popen = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                                 env=env, cwd=work_dir, start_new_session=True)
    except BaseException:
        log_file.close()
        raise

    return Subprocess(popen, log_file)


def checkpoint_info(python_bin: str, script_path: str, model_path: str,
                    env: Dict[str, str], timeout: float = 120.0) -> Optional[Dict]:
    """What a chemprop 2 checkpoint says about itself, or None if it isn't one.

    Only chemprop 2 can open its own checkpoints, so this asks the v2 environment.
    """
    script_path = os.path.abspath(script_path)
    try:
        completed = subprocess.run([python_bin, script_path],
                                   input=json.dumps({'path': model_path}),
                                   capture_output=True, text=True, env=env,
                                   cwd=os.path.dirname(script_path) or '.',
                                   timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None

    if completed.returncode != 0:
        return None

    try:
        info = json.loads(completed.stdout)
    except ValueError:
        return None

    return None if 'error' in info else info


class AttributionWorker:
    """A long-lived chemprop 2 process that answers attribution requests.

    Importing chemprop 2 costs several seconds, which made a fresh process per
    hovered molecule the whole cost of drawing a structure. One worker is kept
    alive instead, restarted if it dies and shut down once idle so it is not
    holding an ensemble in memory indefinitely.
    """

    IDLE_TIMEOUT = 900.0  # seconds before an unused worker is shut down

    def __init__(self, python_bin: str, script_path: str, env: Dict[str, str],
                 stderr_path: Optional[str] = None):
        self.python_bin = python_bin
        self.script_path = script_path
        self.env = env
        self.stderr_path = stderr_path
        self._proc = None
        self._stderr = None
        self._lock = threading.Lock()
        self._last_used = time.monotonic()
        self._reaper_running = False

    # -- lifecycle --------------------------------------------------------
    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _spawn(self) -> None:
        self._stderr = (open(self.stderr_path, 'a') if self.stderr_path
                        else subprocess.DEVNULL)
        self._proc = subprocess.Popen(
            [self.python_bin, self.script_path, '--serve'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
            env=self.env, cwd=os.path.dirname(self.script_path) or '.',
            text=True, bufsize=1, start_new_session=True)

        # One reaper for the life of the worker object: a respawn would otherwise
        # leave the previous one running against the new process.
        if not self._reaper_running:
            self._reaper_running = True
            threading.Thread(target=self._reap_when_idle, daemon=True).start()

    def _reap_when_idle(self) -> None:
        while True:
            time.sleep(60)
            with self._lock:
                if not self._alive():
                    self._reaper_running = False
                    return
                if time.monotonic() - self._last_used > self.IDLE_TIMEOUT:
                    self._kill()
                    self._reaper_running = False
                    return

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        if self._stderr not in (None, subprocess.DEVNULL):
            self._stderr.close()
        self._stderr = None

    def warm(self) -> None:
        """Starts the worker ahead of the first request, in the background."""
        def _start():
            with self._lock:
                if not self._alive():
                    try:
                        self._spawn()
                    except OSError:
                        self._proc = None
        threading.Thread(target=_start, daemon=True).start()

    # -- requests ---------------------------------------------------------
    def request(self, model_paths: Sequence[str], smiles_list: Sequence[str],
                timeout: float):
        """One attribution round trip, or None if the worker could not answer."""
        payload = json.dumps({'model_paths': list(model_paths),
                              'smiles': list(smiles_list)}) + '\n'

        with self._lock:
            self._last_used = time.monotonic()
            if not self._alive():
                try:
                    self._spawn()
                except OSError:
                    return None

            try:
                self._proc.stdin.write(payload)
                self._proc.stdin.flush()
                line = self._read_line(timeout)
            except (BrokenPipeError, OSError):
                line = None

            if not line:
                # A worker that cannot answer is not trusted again; the caller
                # falls back to a one-shot run.
                self._kill()
                return None

            try:
                return json.loads(line).get('weights')
            except ValueError:
                self._kill()
                return None

    def _read_line(self, timeout: float) -> Optional[str]:
        """Reads one response line, giving up rather than blocking forever."""
        selector = selectors.DefaultSelector()
        selector.register(self._proc.stdout, selectors.EVENT_READ)
        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if selector.select(timeout=min(1.0, deadline - time.monotonic())):
                    return self._proc.stdout.readline()
                if not self._alive():
                    return None
            return None
        finally:
            selector.close()


_WORKER = None
_WORKER_LOCK = threading.Lock()


def attribution_worker(python_bin: str, script_path: str, env: Dict[str, str],
                       stderr_path: Optional[str] = None) -> AttributionWorker:
    """The process-wide attribution worker, created on first use."""
    global _WORKER
    with _WORKER_LOCK:
        if _WORKER is None:
            _WORKER = AttributionWorker(python_bin, script_path, env, stderr_path)
        return _WORKER


def atom_weights(python_bin: str, script_path: str, model_paths: Sequence[str],
                 smiles_list: Sequence[str], env: Dict[str, str],
                 stderr_path: Optional[str] = None,
                 timeout: float = 30.0) -> List[Optional[List[float]]]:
    """Per-atom attribution weights from chemprop 2 models.

    Runs ``v2_attribution.py`` under the v2 interpreter, since the weights come
    from the model's own hidden states and only chemprop 2 can load its
    checkpoints. Returns one list per SMILES, or None where the weights could not
    be computed; failures are reported as None rather than raised, because
    attribution is a decoration on a structure that must still be drawn.
    """
    worker = attribution_worker(python_bin, script_path, env, stderr_path)
    weights = worker.request(model_paths, smiles_list, timeout)
    if weights is not None and len(weights) == len(smiles_list):
        return weights

    # The worker could not answer (it had died, or was too slow); fall back to a
    # one-shot run so a structure is still coloured.
    request = json.dumps({'model_paths': list(model_paths), 'smiles': list(smiles_list)})

    # An explicit working directory keeps the chemprop 1.x checkout this app runs
    # from off the helper's import path: the service's working directory is that
    # checkout, and it would otherwise shadow chemprop 2 entirely. (Python's -I
    # would harden this further but also drops the user site-packages that this
    # environment currently resolves some of its dependencies from.)
    try:
        completed = subprocess.run([python_bin, script_path], input=request,
                                   capture_output=True, text=True, env=env,
                                   cwd=os.path.dirname(script_path) or '.',
                                   timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return [None] * len(smiles_list)

    if completed.returncode != 0:
        return [None] * len(smiles_list)

    try:
        weights = json.loads(completed.stdout)['weights']
    except (ValueError, KeyError):
        return [None] * len(smiles_list)

    if len(weights) != len(smiles_list):
        return [None] * len(smiles_list)

    return weights


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


class TrainingProgress:
    """Follows a chemprop 2 run across polls, so the bar advances every epoch.

    Members that have finished are counted exactly, from the model file written
    when each one ends. Within the member training now, the checkpoint Lightning
    rewrites at the end of every epoch is used as a tick: its modification time
    changing means one more epoch is done.

    That tick is the only per-epoch signal a run offers. The metrics file is
    flushed every hundred steps, which on a small dataset is tens of epochs
    apart, and the trainer's progress display writes nothing useful to a file.
    Because ticks can only be counted while watching, the exact records are still
    consulted as a floor, which also covers a bar that started late.
    """

    def __init__(self, output_dir: str, epochs: int, ensemble_size: int = 1):
        self.output_dir = output_dir
        self.epochs = max(1, epochs)
        self.total = self.epochs * max(1, ensemble_size)
        self._member = -1
        self._ticks = 0
        self._last_seen = None

    def poll(self) -> float:
        finished = len(collect_models(self.output_dir))

        if finished != self._member:      # a member ended; start counting the next
            self._member = finished
            self._ticks = 0
            self._last_seen = None

        member_dir = os.path.join(self.output_dir, f'model_{finished}')
        stamp = _last_checkpoint_stamp(member_dir)
        if stamp is not None and stamp != self._last_seen:
            # The first one seen already means an epoch finished.
            self._ticks = self._ticks + 1 if self._last_seen is not None else max(self._ticks, 1)
            self._last_seen = stamp

        within = max(self._ticks, _epochs_in_progress(member_dir))
        # A member counts as complete only once its model file appears.
        within = min(within, self.epochs - 1)

        return min((finished * self.epochs + within) * 100.0 / self.total, 100.0)


def _last_checkpoint_stamp(model_dir: str):
    """Modification time and size of the checkpoint rewritten each epoch."""
    path = os.path.join(model_dir, 'checkpoints', 'last.ckpt')
    try:
        info = os.stat(path)
    except OSError:
        return None
    return (info.st_mtime_ns, info.st_size)


def epoch_progress(output_dir: str, epochs: int, ensemble_size: int = 1) -> float:
    """Percentage of training completed for a chemprop 2 run.

    Ensemble members that have finished are counted exactly, from the model files
    written when each one ends. Progress within the member currently training is
    taken from whichever of its two records is further along: the Lightning
    metrics file, and the epoch named in its best checkpoint. Both lag - the
    metrics file is only flushed every so many rows - but together they keep the
    bar moving, and neither can run ahead of the truth.

    The trainer's own "Epoch N/M" output is deliberately not used: it also prints
    that string outside the progress display, so it cannot be told apart from a
    real reading.
    """
    total = epochs * max(1, ensemble_size)
    if total <= 0:
        return 0.0

    finished = len(collect_models(output_dir))
    done = finished * epochs

    # The member being trained now is the first without a model file.
    current_dir = os.path.join(output_dir, f'model_{finished}')
    if os.path.isdir(current_dir):
        done += min(_epochs_in_progress(current_dir), max(0, epochs - 1))

    return min(done * 100.0 / total, 100.0)


def _epochs_in_progress(model_dir: str) -> int:
    """Epochs finished by one ensemble member, as far as its files show."""
    epochs = 0

    for metrics_csv in glob.glob(os.path.join(model_dir, '**', 'metrics.csv'), recursive=True):
        try:
            with open(metrics_csv) as f:
                seen = {row.get('epoch') for row in csv.DictReader(f)}
            epochs = max(epochs, len({e for e in seen if e not in (None, '')}))
        except OSError:
            continue

    # The best checkpoint names the epoch it came from, and is rewritten whenever
    # the model improves, which is often early on when the bar would otherwise sit
    # at zero.
    for checkpoint in glob.glob(os.path.join(model_dir, 'checkpoints', 'best-epoch=*.ckpt')):
        match = re.search(r'best-epoch=(\d+)', os.path.basename(checkpoint))
        if match:
            epochs = max(epochs, int(match.group(1)) + 1)

    return epochs


# How each backend names its validation metric, and what to call it on the chart.
METRIC_LABELS = {
    'roc': 'AUC', 'auc': 'AUC', 'prc': 'PRC-AUC', 'prc-auc': 'PRC-AUC',
    'accuracy': 'Accuracy', 'f1': 'F1', 'mcc': 'MCC', 'bce': 'cross-entropy',
    'mse': 'MSE', 'rmse': 'RMSE', 'mae': 'MAE', 'r2': 'R²',
}


def metric_label(name: str) -> str:
    """A readable name for a validation metric, e.g. chemprop 2's "roc" is AUC."""
    return METRIC_LABELS.get(name.strip().lower(), name)


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
    return {'metric': metric_label(pretty) if pretty else '', 'models': models}


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
