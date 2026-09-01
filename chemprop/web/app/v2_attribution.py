"""Per-atom attribution weights for chemprop 2 models.

Executed with the chemprop 2 environment's interpreter, never imported by the web
app itself, which runs chemprop 1.x. Reads a JSON request on stdin::

    {"model_paths": ["/path/a.pt", ...], "smiles": ["CCO", ...]}

and writes on stdout::

    {"weights": [[w_atom0, w_atom1, ...] | null, ...]}

one entry per SMILES, averaged over the ensemble, or null where attribution is
not possible.

With ``--serve`` it stays running and answers one request per line instead of
exiting, which is how the web app uses it: importing chemprop 2 costs several
seconds, and paying that per hovered molecule was the whole latency.

The measure is gradient x activation on the per-atom hidden states ("GradCAM"),
the same one the chemprop 1.x path uses, so a structure is coloured on the same
basis whichever backend trained the model.
"""

import json
import sys

import numpy as np
import torch

from chemprop import data, featurizers
from chemprop.models.utils import load_model


def atom_weights(model, smi: str, featurizer):
    """Per-atom weights for one molecule, or None if they can't be computed."""
    datapoint = data.MoleculeDatapoint.from_smi(smi)
    n_atoms = datapoint.mol.GetNumAtoms()
    dataset = data.MoleculeDataset([datapoint], featurizer)
    batch = data.collate_batch([dataset[0]])

    captured = {}

    def capture(_module, _inputs, output):
        # The message passing output is the atom-level representation the
        # attribution is taken over; it is not otherwise reachable from outside.
        output.retain_grad()
        captured['H'] = output

    handle = model.message_passing.register_forward_hook(capture)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            preds = model(batch.bmg, batch.V_d, batch.X_d)
            # Sum over tasks so the gradient reflects overall influence.
            preds.sum().backward()
    finally:
        handle.remove()

    H = captured.get('H')
    if H is None or H.grad is None:
        return None

    weights = (H.grad * H).sum(dim=-1).detach().cpu().numpy()
    if len(weights) < n_atoms:
        return None

    return weights[:n_atoms]


class Models:
    """Loads an ensemble, keeping only the most recent one.

    A served worker would otherwise accumulate every checkpoint it was ever asked
    about, and a CheMeleon ensemble is over 100 MB per model.
    """

    def __init__(self):
        self.paths = None
        self.models = []

    def get(self, model_paths):
        if self.paths == list(model_paths):
            return self.models

        models = []
        for path in model_paths:
            try:
                models.append(load_model(path, multicomponent=False))
            except Exception as e:
                print(f'could not load {path}: {e}', file=sys.stderr)

        self.paths = list(model_paths)
        self.models = models
        return models


def handle(request, cache, featurizer):
    """Answers one attribution request."""
    models = cache.get(request['model_paths'])

    results = []
    for smi in request['smiles']:
        per_model = []
        for model in models:
            try:
                w = atom_weights(model, smi, featurizer)
            except Exception as e:
                print(f'attribution failed for {smi}: {e}', file=sys.stderr)
                w = None
            if w is not None:
                per_model.append(w)
        results.append(np.mean(per_model, axis=0).tolist() if per_model else None)

    return {'weights': results}


def main() -> int:
    serve = '--serve' in sys.argv[1:]
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    cache = Models()

    if not serve:
        json.dump(handle(json.load(sys.stdin), cache, featurizer), sys.stdout)
        return 0

    # One request per line, one response per line, flushed so the caller is not
    # left waiting on a buffer.
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            response = handle(json.loads(line), cache, featurizer)
        except Exception as e:
            print(f'request failed: {e}', file=sys.stderr)
            response = {'weights': None, 'error': str(e)}
        sys.stdout.write(json.dumps(response) + '\n')
        sys.stdout.flush()

    return 0


if __name__ == '__main__':
    sys.exit(main())
