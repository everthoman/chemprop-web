"""Reading model checkpoints without executing what is inside them.

A PyTorch checkpoint is a pickle, and ``torch.load(..., weights_only=False)`` runs
whatever code that pickle asks for. Doing that to a file a user just uploaded
hands them the account this app runs as, so uploads are read here instead, with
``weights_only=True``: torch's restricted unpickler builds tensors and the few
types listed below, and refuses everything else.

The list is what chemprop 1.x checkpoints legitimately contain, established by
loading every checkpoint on this server. It must stay narrow — each entry is a
type an uploaded file is allowed to instantiate.
"""

import argparse
from typing import Any, Dict

import numpy as np
import torch

from chemprop.args import TrainArgs

SAFE_GLOBALS = [
    argparse.Namespace,          # chemprop 1.x stores its training args as one
    np._core.multiarray._reconstruct,
    np._core.multiarray.scalar,
    np.ndarray,
    np.dtype,
    np.dtypes.Float64DType,
    np.dtypes.Float32DType,
    np.dtypes.Int64DType,
    np.dtypes.Int32DType,
    np.dtypes.BoolDType,
    np.dtypes.ObjectDType,
]


def register() -> None:
    """Allows the types above in restricted loads. Safe to call more than once."""
    torch.serialization.add_safe_globals(SAFE_GLOBALS)


def load_checkpoint_dict(path: str) -> Dict[str, Any]:
    """Reads a checkpoint without running code from it.

    :raises Exception: if the file is not a checkpoint, or contains anything
        beyond the allowlisted types — which is what an attacker's file does.
    """
    register()
    return torch.load(path, map_location=lambda storage, loc: storage, weights_only=True)


def load_v1_args(path: str) -> TrainArgs:
    """The training arguments of a chemprop 1.x checkpoint.

    Mirrors ``chemprop.utils.load_args`` but never unpickles arbitrary objects, so
    it can be pointed at a file that has just arrived from a browser.
    """
    data = load_checkpoint_dict(path)
    if not isinstance(data, dict) or 'args' not in data:
        raise ValueError('not a chemprop 1 checkpoint')

    args = TrainArgs()
    args.from_dict(vars(data['args']), skip_unsettable=True)
    return args
