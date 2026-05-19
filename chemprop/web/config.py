"""
Sets the config parameters for the flask app object.
These are accessible in a dictionary, with each line defining a key.
"""

import torch


WEB_VERSION = '1.8.1'

DEFAULT_USER_ID = 1

SMILES_FILENAME = 'smiles.csv'
PREDICTIONS_FILENAME = 'predictions.csv'
DB_FILENAME = 'chemprop.sqlite3'
CUDA = torch.cuda.is_available()
GPUS = list(range(torch.cuda.device_count()))
