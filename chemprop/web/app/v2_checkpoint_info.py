"""Describes a chemprop 2 checkpoint well enough to register it.

Executed with the chemprop 2 environment's interpreter, never imported by the web
app, which runs chemprop 1.x. Reads ``{"path": "/path/to/model.pt"}`` on stdin and
writes on stdout::

    {"n_tasks": 1, "task_type": "regression", "multicomponent": false}

or ``{"error": "..."}`` when the file is not a chemprop 2 model. A checkpoint
records how many targets it predicts and what kind of task they are, but not what
those targets are called, so the names are asked for at upload time.
"""

import json
import sys

from chemprop.models.utils import load_model

# chemprop 2 expresses the task through the predictor class rather than a stored
# label; these are the prefixes it uses for each family of head.
TASK_TYPES = (
    ('Multiclass', 'multiclass'),
    ('Binary', 'classification'),
    ('Classification', 'classification'),
    ('Spectral', 'spectral'),
    ('Regression', 'regression'),
    ('Mve', 'regression'),
    ('Evidential', 'regression'),
    ('Quantile', 'regression'),
)


def task_type_of(predictor) -> str:
    name = type(predictor).__name__
    for prefix, task_type in TASK_TYPES:
        if name.startswith(prefix):
            return task_type
    return 'regression'


def main() -> int:
    path = json.load(sys.stdin)['path']

    multicomponent = False
    try:
        model = load_model(path, multicomponent=False)
    except Exception:
        try:
            model = load_model(path, multicomponent=True)
            multicomponent = True
        except Exception as e:
            json.dump({'error': f'{type(e).__name__}: {e}'}, sys.stdout)
            return 0

    json.dump({
        'n_tasks': int(model.predictor.n_tasks),
        'task_type': task_type_of(model.predictor),
        'multicomponent': multicomponent,
    }, sys.stdout)
    return 0


if __name__ == '__main__':
    sys.exit(main())
