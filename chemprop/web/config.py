"""
Sets the config parameters for the flask app object.
These are accessible in a dictionary, with each line defining a key.
"""

import torch


WEB_VERSION = '1.8.7'

DEFAULT_USER_ID = 1

SMILES_FILENAME = 'smiles.csv'
PREDICTIONS_FILENAME = 'predictions.csv'
TRAIN_TEST_PREDS_FILENAME = 'train_test_predictions.csv'
DB_FILENAME = 'chemprop.sqlite3'
CUDA = torch.cuda.is_available()
GPUS = list(range(torch.cuda.device_count()))
