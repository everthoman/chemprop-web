"""
Sets the config parameters for the flask app object.
These are accessible in a dictionary, with each line defining a key.
"""

import os
import shutil
import subprocess
from datetime import timedelta

import torch


WEB_VERSION = '1.8.8'

DEFAULT_USER_ID = 1

# Usernames with admin rights (e.g. creating new users). Comma-separated, case-insensitive.
ADMIN_USERS = [u.strip().lower() for u in os.environ.get('CHEMPROP_ADMIN_USERS', 'evehom').split(',') if u.strip()]

SMILES_FILENAME = 'smiles.csv'
PREDICTIONS_FILENAME = 'predictions.csv'
DB_FILENAME = 'chemprop.sqlite3'
CREDENTIALS_FILENAME = 'users_auth.json'  # JSON map of username -> password hash (no database)
SECRET_KEY_FILENAME = '.flask_secret_key'  # persisted key used to sign session cookies
PERMANENT_SESSION_LIFETIME = timedelta(days=30)
# CUDA numbers devices fastest-first unless told otherwise, while nvidia-smi
# numbers them by PCI bus. On a host whose fastest card is not the first on the
# bus the two disagree, and the index chosen in the dropdown would then select a
# different card than its label names. Pin the order so both agree everywhere.
# Set before any CUDA call: the driver reads it when it initialises.
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')

CUDA = torch.cuda.is_available()

# Cards too small to train on are left out of the dropdown. This host has a 2 GB
# display adapter beside its two training cards, and choosing it only produces an
# out-of-memory failure part way into a run.
MIN_GPU_MEMORY_GB = float(os.environ.get('CHEMPROP_MIN_GPU_MEMORY_GB', '4'))


def _usable_gpus():
    """The GPUs worth offering, as ``(indices, {index: label})``.

    Asked of nvidia-smi rather than torch: reading device properties would
    initialise CUDA in this process, which then forks the progress-bar
    subprocess, and CUDA does not survive a fork.
    """
    try:
        listing = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        listing = None

    if listing is None or listing.returncode != 0:
        # Without nvidia-smi, offer every card rather than none.
        return list(range(torch.cuda.device_count())), {}

    # A restricted set renumbers the devices: with CUDA_VISIBLE_DEVICES=1,2 the
    # training process calls them 0 and 1, whatever nvidia-smi calls them.
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    allowed = None
    if visible is not None:
        allowed = [int(part) for part in visible.split(',') if part.strip().isdigit()]

    indices, labels = [], {}
    for line in listing.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) < 3:
            continue
        try:
            physical, memory_gb = int(parts[0]), float(parts[2]) / 1024
        except ValueError:
            continue

        if allowed is not None:
            if physical not in allowed:
                continue
            index = allowed.index(physical)
        else:
            index = physical

        if memory_gb < MIN_GPU_MEMORY_GB:
            continue

        indices.append(index)
        labels[index] = f'{index} — {parts[1]} ({memory_gb:.0f} GB)'

    return sorted(indices), labels


GPUS, GPU_LABELS = _usable_gpus()

# --- chemprop 2.x backend -------------------------------------------------
# Foundation models (e.g. CheMeleon) are chemprop 2.x artifacts and cannot be
# loaded by the chemprop 1.x code this app runs on. They are used by shelling
# out to the chemprop CLI installed in a separate conda environment, so this
# process never imports chemprop 2.x.
CHEMPROP2_ENV = os.environ.get('CHEMPROP2_ENV', 'chemprop2')
CONDA_EXE = os.environ.get('CONDA_EXE') or shutil.which('conda') or \
    os.path.expanduser('~/Programs/miniconda3/bin/conda')

# The environment's own entry point, invoked directly rather than through
# `conda run`: that wrapper puts two extra processes between this app and the
# training process, and they land in a different process group, so cancelling a
# run killed the wrapper and left the training itself running on the GPU.
CHEMPROP2_BIN = os.environ.get('CHEMPROP2_BIN') or os.path.join(
    os.path.dirname(os.path.dirname(CONDA_EXE)), 'envs', CHEMPROP2_ENV, 'bin', 'chemprop')

# The environment's interpreter, used to run helper scripts (atom attribution)
# that need the chemprop 2 API rather than its CLI.
CHEMPROP2_PYTHON = os.path.join(os.path.dirname(CHEMPROP2_BIN), 'python')

# The v2 backend is offered on the Train page only when its environment is present.
CHEMPROP2_AVAILABLE = os.path.isfile(CHEMPROP2_BIN)

# Foundation models selectable when training with the v2 backend. The names are
# passed straight to `chemprop train --from-foundation`.
FOUNDATION_MODELS = ['CheMeleon']

# Default batch size offered for v2 training. Kept at chemprop 2's own default:
# raising it was measured to buy no wall-clock time (CheMeleon finetuning is
# GPU-bound at 64 already) while costing validation loss at a fixed epoch budget.
CHEMPROP2_BATCH_SIZE = 64

# Search-space keywords the Hyperopt page may pass through. They land in a
# variadic CLI option, so anything not on this list is dropped rather than
# becoming an argument of its own.
SEARCH_KEYWORDS = ['basic', 'learning_rate', 'init_lr', 'final_lr', 'all',
                   'linked_hidden_size', 'hidden_size', 'ffn_hidden_size',
                   'ffn_num_layers', 'depth', 'dropout', 'batch_size',
                   'warmup_epochs', 'max_lr', 'aggregation', 'activation']
