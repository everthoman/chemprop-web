"""Defines a number of routes/views for the flask app."""

from functools import wraps
import csv
import io
import math
import os
import re
import sys
import shutil
from tempfile import TemporaryDirectory, NamedTemporaryFile
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
from chemprop.data import get_data, get_header, get_smiles, get_task_names, validate_data, split_data
from chemprop.train import make_predictions, run_training
from chemprop.utils import create_logger, load_task_names, load_args
from chemprop.web.app.atom_attribution import compute_attributions
from chemprop.web.app.applicability import ApplicabilityDomain, load_training_smiles

TRAINING = 0
TRAINING_MODE = ''  # 'train', 'hyperopt', or 'predict'
PROGRESS = mp.Value('d', 0.0)
CURRENT_LOG_PATH = ''
ACTIVE_PROCESS = None
PROGRESS_BAR_PROCESS = None
CANCELLED = False
LAST_TRAIN_RESULT = {}  # user_key -> last successful training kwargs, served on GET after tab switch


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
    defaulting to 0.1 (i.e. 90% target coverage).
    """
    enabled = form.get('conformalEnabled', 'False') == 'True'
    raw = (form.get('conformalAlpha', '') or '').strip()
    try:
        alpha = float(raw) if raw else 0.1
    except ValueError:
        alpha = 0.1
    if not (0 < alpha < 1):
        alpha = 0.1
    return enabled, alpha


def conformal_calibration_feasible(n_cal: int, alpha: float) -> bool:
    """Whether a calibration set of size ``n_cal`` can yield a finite conformal quantile.

    chemprop computes ``q_level = ceil((n+1)(1-alpha)) / n`` and takes that quantile of
    the calibration scores; it must be <= 1 for the quantile to be well defined.
    """
    return n_cal >= 1 and math.ceil((n_cal + 1) * (1 - alpha)) <= n_cal


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


@app.route('/receiver', methods=['POST'])
@check_not_demo
def receiver():
    """Receiver monitoring the progress of training."""
    val_curves = {}
    if CURRENT_LOG_PATH and os.path.exists(CURRENT_LOG_PATH):
        val_curves = _parse_val_curves(CURRENT_LOG_PATH)
    return jsonify(progress=PROGRESS.value, training=TRAINING, mode=TRAINING_MODE, val_curves=val_curves)


@app.route('/cancel', methods=['POST'])
def cancel():
    """Terminates any active training, hyperopt, or prediction subprocess."""
    global ACTIVE_PROCESS, PROGRESS_BAR_PROCESS, CANCELLED, TRAINING, TRAINING_MODE, PROGRESS, CURRENT_LOG_PATH
    for proc in [ACTIVE_PROCESS, PROGRESS_BAR_PROCESS]:
        if proc and proc.is_alive():
            proc.terminate()
            proc.join(timeout=3)
            if proc.is_alive():
                proc.kill()
    ACTIVE_PROCESS = None
    PROGRESS_BAR_PROCESS = None
    CANCELLED = True
    TRAINING = 0
    TRAINING_MODE = ''
    PROGRESS = mp.Value('d', 0.0)
    CURRENT_LOG_PATH = ''
    return jsonify(success=True)


# Endpoints reachable without being logged in.
PUBLIC_ENDPOINTS = {'login', 'static'}


def current_user_id() -> Optional[int]:
    """The authenticated user's id, read from the signed session.

    This replaces the legacy ``currentUser`` cookie as the source of identity,
    so a client can no longer reach another user's data by editing a cookie.
    """
    return session.get('user_id')


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

    return render_template('train.html',
                           datasets=db.get_datasets(current_user_id()),
                           current_user=current_user_id() or '',
                           cuda=app.config['CUDA'],
                           gpus=app.config['GPUS'],
                           data_upload_warnings=data_upload_warnings,
                           data_upload_errors=data_upload_errors,
                           users=db.get_all_users(),
                           **kwargs)


@app.route('/train', methods=['GET', 'POST'])
@check_not_demo
def train():
    """Renders the train page and performs training if request method is POST."""
    global PROGRESS, TRAINING, TRAINING_MODE, CURRENT_LOG_PATH, ACTIVE_PROCESS, PROGRESS_BAR_PROCESS, CANCELLED

    warnings, errors = [], []

    if request.method == 'GET':
        return render_train()

    # Get arguments
    data_name, epochs, ensemble_size, checkpoint_name = \
        request.form['dataName'], int(request.form['epochs']), \
        int(request.form['ensembleSize']), request.form['checkpointName']
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
    use_progress_bar = request.form.get('useProgressBar', 'True') == 'True'
    conformal_enabled, conformal_alpha = parse_conformal_form(request.form)

    # Handle optional hyperopt config (content sent as hidden field via FileReader)
    config_content = request.form.get('configFileContent', '').strip()
    config_path = None
    if config_content:
        config_path = os.path.join(app.config['TEMP_FOLDER'], 'uploaded_config.json')
        with open(config_path, 'w') as f:
            f.write(config_content)

    # Create and modify args
    train_arg_list = [
        '--data_path', data_path,
        '--dataset_type', dataset_type,
        '--epochs', str(epochs),
        '--ensemble_size', str(ensemble_size),
        '--split_type', split_type,
    ]
    if config_path is not None:
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
        errors.append('Selected classification dataset but not all labels are 0 or 1. Select regression instead.')

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
                                        len(targets))

    with TemporaryDirectory() as temp_dir:
        args.save_dir = temp_dir

        LAST_TRAIN_RESULT.pop(str(current_user), None)

        if use_progress_bar:
            pb_proc = mp.Process(target=progress_bar, args=(args, PROGRESS))
            pb_proc.start()
            PROGRESS_BAR_PROCESS = pb_proc
            TRAINING = 1
            TRAINING_MODE = 'train'

        CURRENT_LOG_PATH = os.path.join(temp_dir, 'verbose.log')
        train_proc = _spawn.Process(target=_train_worker,
                                    args=(train_arg_list, args.task_names, data_path,
                                          ignore_cols, id_col, temp_dir, patience, min_delta))
        train_proc.start()
        ACTIVE_PROCESS = train_proc

        while train_proc.is_alive():
            train_proc.join(timeout=0.5)

        ACTIVE_PROCESS = None
        log_path = CURRENT_LOG_PATH  # snapshot before cancel() might clear it
        cancelled = CANCELLED
        CANCELLED = False

        # Only treat as cancelled if the process was actually killed (negative
        # exitcode). A cancel click that arrived after natural completion should
        # not suppress the training results.
        was_killed = cancelled and train_proc.exitcode is not None and train_proc.exitcode < 0

        if was_killed or train_proc.exitcode not in (0, None):
            if use_progress_bar:
                if pb_proc.is_alive():
                    pb_proc.terminate()
                    pb_proc.join(timeout=2)
                PROGRESS_BAR_PROCESS = None
                TRAINING = 0
                TRAINING_MODE = ''
                PROGRESS = mp.Value('d', 0.0)
            CURRENT_LOG_PATH = ''
            if was_killed:
                return render_train(warnings=['Training was cancelled.'])
            errors.append('Training failed — check server logs for details.')
            return render_train(warnings=warnings, errors=errors)

        if use_progress_bar:
            PROGRESS.value = 100
            pb_proc.join()
            PROGRESS_BAR_PROCESS = None
            # Keep TRAINING=1 until results are fully prepared so the polling JS
            # doesn't reload the page before LAST_TRAIN_RESULT is populated.

        # Parse convergence data before temp_dir is cleaned up
        val_curves = _parse_val_curves(log_path)
        CURRENT_LOG_PATH = ''

        # Check if name overlap
        if checkpoint_name != ckpt_name:
            warnings.append(name_already_exists_message('Checkpoint', checkpoint_name, ckpt_name))

        # Move models
        for root, _, files in os.walk(args.save_dir):
            for fname in files:
                if fname.endswith('.pt'):
                    model_id = db.insert_model(ckpt_id)
                    save_path = os.path.join(app.config['CHECKPOINT_FOLDER'], f'{model_id}.pt')
                    shutil.move(os.path.join(args.save_dir, root, fname), save_path)

    # Generate visualization data
    plot_data = None
    conformal_info = None
    if dataset_type in ['regression', 'classification']:
        try:
            model_paths = [os.path.join(app.config['CHECKPOINT_FOLDER'], f'{m["id"]}.pt')
                           for m in db.get_models(ckpt_id)]

            # Reload data fresh from CSV — run_training scales targets in-place,
            # so reusing `data` here would give scaled targets against unscaled predictions.
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
            pred_arguments = [
                '--test_path', 'None',
                '--preds_path', os.path.join(app.config['TEMP_FOLDER'], 'train_plot_preds.csv'),
                '--checkpoint_paths', *model_paths,
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

            if dataset_type == 'regression':
                if _use_unc:
                    train_preds, train_unc = make_predictions(args=pred_args, smiles=to_smiles_list(train_split), return_uncertainty=True)
                    test_preds, test_unc = make_predictions(args=pred_args, smiles=to_smiles_list(test_split), return_uncertainty=True)
                    train_std = [_tt_var_to_std(r) for r in train_unc]
                    test_std  = [_tt_var_to_std(r) for r in test_unc]
                else:
                    train_preds = make_predictions(args=pred_args, smiles=to_smiles_list(train_split), return_uncertainty=False)
                    test_preds = make_predictions(args=pred_args, smiles=to_smiles_list(test_split), return_uncertainty=False)
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

                # Write combined train/test CSV for download
                tt_path = os.path.join(app.config['TEMP_FOLDER'], app.config['TRAIN_TEST_PREDS_FILENAME'])
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
                    train_preds, train_unc = make_predictions(args=pred_args, smiles=to_smiles_list(train_split), return_uncertainty=True)
                    test_preds, test_unc = make_predictions(args=pred_args, smiles=to_smiles_list(test_split), return_uncertainty=True)
                    train_std = [_tt_var_to_std(r) for r in train_unc]
                    test_std  = [_tt_var_to_std(r) for r in test_unc]
                else:
                    train_preds = make_predictions(args=pred_args, smiles=to_smiles_list(train_split), return_uncertainty=False)
                    test_preds = make_predictions(args=pred_args, smiles=to_smiles_list(test_split), return_uncertainty=False)
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

                # Write combined train/test CSV for download
                tt_path = os.path.join(app.config['TEMP_FOLDER'], app.config['TRAIN_TEST_PREDS_FILENAME'])
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

            # Conformal prediction: save the held-out validation split as the
            # calibration set so predictions can carry conformal intervals/sets,
            # and report empirical coverage on the independent test split.
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

                    cal_method = 'conformal_regression' if dataset_type == 'regression' else 'conformal'
                    conf_arguments = [
                        '--test_path', 'None',
                        '--preds_path', os.path.join(app.config['TEMP_FOLDER'], 'train_conformal_preds.csv'),
                        '--checkpoint_paths', *model_paths,
                        # See note in predict(): name the calibration SMILES column so the
                        # in-memory-SMILES code path can still load the calibration CSV.
                        '--smiles_columns', 'smiles',
                        '--calibration_path', cal_path,
                        '--calibration_method', cal_method,
                        '--conformal_alpha', str(conformal_alpha),
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
                    n_tasks = len(args.task_names)

                    per_task = []
                    if dataset_type == 'regression':
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
                    else:  # classification: in_set / out_set indicators per task
                        for ti, name in enumerate(args.task_names):
                            total = 0
                            confident_correct = confident_total = 0
                            cats = {'positive': 0, 'uncertain': 0, 'negative': 0}
                            for j in range(len(conf_preds)):
                                yt = test_targets_c[j][ti]
                                if yt is None:
                                    continue
                                try:
                                    in_set = int(round(float(conf_unc[j][ti])))
                                    out_set = int(round(float(conf_unc[j][ti + n_tasks])))
                                except (TypeError, ValueError, IndexError):
                                    continue
                                total += 1
                                label = int(yt)
                                if in_set == 1:
                                    cats['positive'] += 1
                                    confident_total += 1
                                    confident_correct += (label == 1)
                                elif out_set == 0:
                                    cats['negative'] += 1
                                    confident_total += 1
                                    confident_correct += (label == 0)
                                else:
                                    cats['uncertain'] += 1
                            per_task.append({
                                'name': name,
                                'categories': cats,
                                'confident_accuracy': round(confident_correct / confident_total, 3) if confident_total else None,
                                'n': total,
                            })

                    conformal_info = {
                        'enabled': True,
                        'alpha': conformal_alpha,
                        'n_cal': n_cal,
                        'mode': dataset_type,
                        'per_task': per_task,
                    }

        except Exception as e:
            warnings.append(f'Could not generate visualization: {str(e)}')

    tt_path = os.path.join(app.config['TEMP_FOLDER'], app.config['TRAIN_TEST_PREDS_FILENAME'])
    LAST_TRAIN_RESULT[str(current_user)] = dict(
        trained=True,
        dataset_type=dataset_type,
        plot_data=plot_data,
        ckpt_id=ckpt_id,
        val_curves=val_curves,
        conformal=conformal_info,
        warnings=warnings,
        errors=errors,
    )

    # Persist results to disk so the Checkpoints page can show them permanently
    results_path = os.path.join(app.config['CHECKPOINT_FOLDER'], f'{ckpt_id}_results.json')
    try:
        class _NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        with open(results_path, 'w') as _f:
            json.dump({'dataset_type': dataset_type, 'plot_data': plot_data,
                       'val_curves': val_curves, 'conformal': conformal_info}, _f, cls=_NumpyEncoder)
    except Exception as e:
        warnings.append(f'Could not save results for Checkpoints page: {str(e)}')

    # Copy train/test predictions CSV to permanent storage alongside the checkpoint
    if os.path.exists(tt_path):
        preds_dest = os.path.join(app.config['CHECKPOINT_FOLDER'], f'{ckpt_id}_train_test_preds.csv')
        try:
            shutil.copy2(tt_path, preds_dest)
        except Exception:
            pass

    # Now that LAST_TRAIN_RESULT and the JSON file are ready, allow the polling
    # JS to detect training completion and reload to show results.
    if use_progress_bar:
        TRAINING = 0
        TRAINING_MODE = ''
        PROGRESS = mp.Value('d', 0.0)

    preds_perm_path = os.path.join(app.config['CHECKPOINT_FOLDER'], f'{ckpt_id}_train_test_preds.csv')
    return render_train(trained=True,
                        dataset_type=dataset_type,
                        plot_data=plot_data,
                        ckpt_id=ckpt_id,
                        val_curves=val_curves,
                        conformal=conformal_info,
                        train_test_preds_available=os.path.exists(preds_perm_path),
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


def render_hyperopt(**kwargs):
    """Renders the hyperopt page with specified kwargs."""
    data_upload_warnings, data_upload_errors = get_upload_warnings_errors('data')
    return render_template('hyperopt.html',
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
    global PROGRESS, TRAINING, TRAINING_MODE, ACTIVE_PROCESS, PROGRESS_BAR_PROCESS, CANCELLED

    warnings, errors = [], []

    if request.method == 'GET':
        return render_hyperopt()

    # Get form fields
    data_name = request.form['dataName']
    epochs = int(request.form['epochs'])
    num_iters = int(request.form['numIters'])
    dataset_type = request.form.get('datasetType', 'regression')
    gpu = request.form.get('gpu')
    search_keywords = request.form.getlist('searchKeywords') or ['basic']
    id_col = request.form.get('idColumn', '').strip() or None

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

        pb_proc = mp.Process(target=hyperopt_progress_bar,
                             args=(hyper_args.hyperopt_checkpoint_dir, num_iters, PROGRESS))
        pb_proc.start()
        PROGRESS_BAR_PROCESS = pb_proc
        TRAINING = 1
        TRAINING_MODE = 'hyperopt'

        hyper_proc = _spawn.Process(target=_hyperopt_worker, args=(hyper_args_list,))
        hyper_proc.start()
        ACTIVE_PROCESS = hyper_proc

        while hyper_proc.is_alive():
            hyper_proc.join(timeout=0.5)

        ACTIVE_PROCESS = None
        cancelled = CANCELLED
        CANCELLED = False

        if cancelled or hyper_proc.exitcode != 0:
            if pb_proc.is_alive():
                pb_proc.terminate()
                pb_proc.join(timeout=2)
            PROGRESS_BAR_PROCESS = None
            TRAINING = 0
            TRAINING_MODE = ''
            PROGRESS = mp.Value('d', 0.0)
            if cancelled:
                return render_hyperopt(warnings=['Hyperopt was cancelled.'])
            errors.append('Hyperopt failed — check server logs for details.')
            return render_hyperopt(warnings=warnings, errors=errors)

        PROGRESS.value = 100
        pb_proc.join()
        PROGRESS_BAR_PROCESS = None
        TRAINING = 0
        TRAINING_MODE = ''
        PROGRESS = mp.Value('d', 0.0)

    # Load best hyperparams from saved config
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


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Renders the predict page and makes predictions if the method is POST."""
    if request.method == 'GET':
        return render_predict()

    # Get arguments
    ckpt_id = request.form['checkpointName']

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

    task_names = load_task_names(model_paths[0])
    num_tasks = len(task_names)
    gpu = request.form.get('gpu')
    train_args = load_args(model_paths[0])

    # Conformal prediction (optional): needs a calibration set saved at train time
    # and a regression/classification model. Falls back to standard output if either
    # is missing, surfacing a warning to the user.
    conformal_enabled, conformal_alpha = parse_conformal_form(request.form)
    cal_path = conformal_calibration_path(ckpt_id)
    pred_warnings = []
    use_conformal = False
    if conformal_enabled:
        if train_args.dataset_type not in ('regression', 'classification'):
            pred_warnings.append('Conformal prediction is not available for this model type; showing standard predictions.')
        elif not os.path.exists(cal_path):
            pred_warnings.append('Conformal prediction is unavailable for this checkpoint — no calibration set was saved when it was trained. Showing standard predictions.')
        else:
            use_conformal = True

    # Build arguments. Conformal and ensemble std-dev are mutually exclusive output
    # modes (chemprop rejects an uncertainty method alongside conformal regression);
    # when conformal is active it takes precedence.
    ensemble_unc = len(model_paths) > 1 and not use_conformal
    return_unc = ensemble_unc or use_conformal
    arguments = [
        '--test_path', 'None',
        '--preds_path', os.path.join(app.config['TEMP_FOLDER'], app.config['PREDICTIONS_FILENAME']),
        '--checkpoint_paths', *model_paths
    ]
    if use_conformal:
        cal_method = 'conformal_regression' if train_args.dataset_type == 'regression' else 'conformal'
        arguments += [
            # Name the calibration file's SMILES column explicitly: when predicting from
            # an in-memory SMILES list there is no test CSV to auto-detect it from, so
            # smiles_columns would otherwise be [None] and the calibration load would fail.
            '--smiles_columns', 'smiles',
            '--calibration_path', cal_path,
            '--calibration_method', cal_method,
            '--conformal_alpha', str(conformal_alpha),
        ]
    elif ensemble_unc:
        arguments += ['--uncertainty_method', 'ensemble']

    if gpu is not None:
        if gpu == 'None':
            arguments.append('--no_cuda')
        else:
            arguments += ['--gpu', gpu]

    # Handle additional features
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

    # Parse arguments
    global ACTIVE_PROCESS, CANCELLED, TRAINING, TRAINING_MODE

    # Run predictions in a subprocess so they can be cancelled
    result_queue = _spawn.Queue()
    pred_proc = _spawn.Process(target=_predict_worker, args=(arguments, smiles, result_queue, return_unc))
    pred_proc.start()
    ACTIVE_PROCESS = pred_proc
    TRAINING = 1
    TRAINING_MODE = 'predict'

    while pred_proc.is_alive():
        pred_proc.join(timeout=0.5)

    ACTIVE_PROCESS = None
    TRAINING = 0
    TRAINING_MODE = ''
    cancelled = CANCELLED
    CANCELLED = False

    if cancelled:
        return render_predict(warnings=['Prediction was cancelled.'])
    if pred_proc.exitcode != 0:
        return render_predict(errors=['Prediction failed — check server logs for details.'])

    result = result_queue.get_nowait()
    if not result['success']:
        return render_predict(errors=[result['error']])
    preds = result['preds']
    raw_unc = result.get('unc')

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
    if use_conformal and raw_unc is not None:
        if train_args.dataset_type == 'regression':
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
        else:
            # raw_unc[i] = [in_set_t1..in_set_tn, out_set_t1..out_set_tn] (0/1 indicators).
            categories = []
            for i in range(len(preds)):
                row = raw_unc[i] if raw_unc[i] is not None else []
                cell = []
                for ti in range(num_tasks):
                    try:
                        in_set = int(round(float(row[ti])))
                        out_set = int(round(float(row[ti + num_tasks])))
                        if in_set == 1:
                            cat = 'positive'
                        elif out_set == 0:
                            cat = 'negative'
                        else:
                            cat = 'uncertain'
                        cell.append({'cat': cat, 'in_set': in_set, 'out_set': out_set})
                    except (TypeError, ValueError, IndexError):
                        cell.append(None)
                categories.append(cell)
            conformal_result = {'mode': 'classification', 'alpha': conformal_alpha, 'categories': categories}

    # Replace invalid smiles with message
    invalid_smiles_warning = 'Invalid SMILES String'
    preds = [pred if pred is not None else [invalid_smiles_warning] * num_tasks for pred in preds]

    # Compute atom attribution SVGs (best-effort; failures yield None)
    flat_smiles = [s[0] for s in smiles[:10]]
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
                    conf_cols += [f'conformal_{t}', f'in_set_{t}', f'out_set_{t}']
        ad_cols = ['ad_similarity', 'ad_in_domain'] if applicability is not None else []
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
                        row.extend(['', '', ''] if cell is None else [cell['cat'], cell['in_set'], cell['out_set']])
            if applicability is not None:
                ad = applicability[idx]
                if ad is not None:
                    row.extend([ad['similarity'], 'yes' if ad['in_domain'] else 'no'])
                else:
                    row.extend(['', ''])
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

    models = db.get_models(ckpt_id)
    if not models:
        return jsonify({'error': 'No models found'}), 404

    model_paths = [os.path.join(app.config['CHECKPOINT_FOLDER'], f'{m["id"]}.pt') for m in models]

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
    """Downloads the best hyperparameter config from the last hyperopt run as a .json file."""
    return send_from_directory(app.config['TEMP_FOLDER'], 'hyperopt_config.json', as_attachment=True,
                               download_name='hyperopt_config.json', cache_timeout=-1)


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
    row = db.query_db('SELECT dataset_name FROM dataset WHERE id = ?', (dataset,), one=True)
    download_name = f'{row["dataset_name"]}.csv' if row else f'{dataset}.csv'
    return send_from_directory(app.config['DATA_FOLDER'], f'{dataset}.csv', as_attachment=True,
                               download_name=download_name, cache_timeout=-1)


@app.route('/data/delete/<int:dataset>')
@check_not_demo
def delete_data(dataset: int):
    """
    Deletes a dataset.

    :param dataset: The id of the dataset to delete.
    """
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
    results_path = os.path.join(app.config['CHECKPOINT_FOLDER'], f'{ckpt_id}_results.json')
    if not os.path.exists(results_path):
        return jsonify(error='No results saved for this checkpoint'), 404
    with open(results_path) as _f:
        data = json.load(_f)
    preds_path = os.path.join(app.config['CHECKPOINT_FOLDER'], f'{ckpt_id}_train_test_preds.csv')
    data['has_preds_csv'] = os.path.exists(preds_path)
    return jsonify(data)


@app.route('/checkpoint/<int:ckpt_id>/download_predictions')
@check_not_demo
def download_checkpoint_predictions(ckpt_id: int):
    """Downloads the permanent train/test predictions CSV for a checkpoint."""
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
        ckpt_args = load_args(ckpt_paths[0])
        ckpt_id, new_ckpt_name = db.insert_ckpt(ckpt_name,
                                                current_user,
                                                ckpt_args.dataset_type,
                                                ckpt_args.epochs,
                                                len(ckpt_paths),
                                                ckpt_args.train_data_size)

        for ckpt_path in ckpt_paths:
            model_id = db.insert_model(ckpt_id)
            model_path = os.path.join(app.config['CHECKPOINT_FOLDER'], f'{model_id}.pt')

            if ckpt_name != new_ckpt_name:
                warnings.append(name_already_exists_message('Checkpoint', ckpt_name, new_ckpt_name))

            shutil.copy(ckpt_path, model_path)

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
    ckpt = db.query_db('SELECT * FROM ckpt WHERE id = ?', (checkpoint,), one=True)
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
    for suffix in ('_results.json', '_train_test_preds.csv', '_calibration.csv'):
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
    new_name = request.form.get('name', '').strip()
    if not new_name:
        return jsonify(success=False, error='Name cannot be empty.')
    try:
        db.rename_ckpt(checkpoint, new_name)
        return jsonify(success=True, name=new_name)
    except Exception:
        return jsonify(success=False, error='That name is already in use.')
