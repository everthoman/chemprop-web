"""
Sets the config parameters for the flask app object.
These are accessible in a dictionary, with each line defining a key.
"""

import os
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
