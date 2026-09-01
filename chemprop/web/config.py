"""
Sets the config parameters for the flask app object.
These are accessible in a dictionary, with each line defining a key.
"""

import os
import shutil
from datetime import timedelta

import torch


WEB_VERSION = '1.8.8'

DEFAULT_USER_ID = 1

# Usernames with admin rights (e.g. creating new users). Comma-separated, case-insensitive.
ADMIN_USERS = [u.strip().lower() for u in os.environ.get('CHEMPROP_ADMIN_USERS', 'evehom').split(',') if u.strip()]

SMILES_FILENAME = 'smiles.csv'
PREDICTIONS_FILENAME = 'predictions.csv'
TRAIN_TEST_PREDS_FILENAME = 'train_test_predictions.csv'
DB_FILENAME = 'chemprop.sqlite3'
CREDENTIALS_FILENAME = 'users_auth.json'  # JSON map of username -> password hash (no database)
SECRET_KEY_FILENAME = '.flask_secret_key'  # persisted key used to sign session cookies
PERMANENT_SESSION_LIFETIME = timedelta(days=30)
CUDA = torch.cuda.is_available()
GPUS = list(range(torch.cuda.device_count()))

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
