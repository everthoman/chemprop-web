"""Subprocess worker functions for training, hyperopt, and prediction.

These must live in a separate module so that mp.get_context('spawn').Process
can import them without triggering Flask/app initialization.
All chemprop imports are deferred to inside each function so that the module
itself is lightweight to import.
"""


def train_worker(train_arg_list, task_names, data_path, ignore_cols, id_col, save_dir):
    import logging as _logging
    from chemprop.args import TrainArgs
    from chemprop.data import get_data
    from chemprop.train import run_training
    from chemprop.utils import create_logger
    from chemprop.constants import TRAIN_LOGGER_NAME

    args = TrainArgs().parse_args(train_arg_list + ['--save_dir', save_dir])
    args.task_names = task_names
    data = get_data(path=data_path, smiles_columns=args.smiles_columns,
                    ignore_columns=ignore_cols or None, store_row=bool(id_col))
    if TRAIN_LOGGER_NAME in _logging.root.manager.loggerDict:
        _logging.getLogger(TRAIN_LOGGER_NAME).handlers.clear()
        del _logging.root.manager.loggerDict[TRAIN_LOGGER_NAME]
    logger = create_logger(name=TRAIN_LOGGER_NAME, save_dir=save_dir, quiet=args.quiet)
    run_training(args, data, logger)


def hyperopt_worker(hyper_args_list):
    from chemprop.args import HyperoptArgs
    from chemprop.hyperparameter_optimization import hyperopt as run_hyperopt
    hyper_args = HyperoptArgs().parse_args(hyper_args_list)
    run_hyperopt(hyper_args)


def predict_worker(arguments, smiles, result_queue):
    from chemprop.args import PredictArgs
    from chemprop.train import make_predictions
    try:
        args = PredictArgs().parse_args(arguments)
        preds = make_predictions(args=args, smiles=smiles, return_uncertainty=False)
        result_queue.put({'success': True, 'preds': preds})
    except Exception as e:
        result_queue.put({'success': False, 'error': str(e)})
