from typing import List

from fedot.api.fedot_cli import create_parser, separate_argparse_to_fedot, preprocess_keys, run_fedot, \
    arguments_dicts
from fedot.core.utils import fedot_project_root


def call_cli_with_parameters(call_string: List[str]):
    """ Function that imitates argparse api call"""
    parser = create_parser(arguments_dicts)
    parameters = parser.parse_args(call_string)
    main_params, fit_params = separate_argparse_to_fedot(parameters)
    preprocess_keys(main_params)
    preprocess_keys(fit_params)
    predictions = run_fedot(parameters, main_params, fit_params, path_to_save=None)
    return predictions


def test_cli_with_parameters():
    """ Test all parameters used in cli are available from api"""
    project_root_path = fedot_project_root()
    ts_train_path = project_root_path.joinpath('test/data/simple_time_series.csv')
    ts_call = (
        f'--problem ts_forecasting --preset fast_train --timeout 0.1 --depth 3 --arity 3 '
        '--popsize 3 --gen_num 5 --opers lagged linear ridge --tuning 0 '
        f'--cv_folds 2 --target sea_height --train {ts_train_path} '
        f'--test {ts_train_path} --for_len 10'
    ).split()
    class_train_path = project_root_path.joinpath('test/data/simple_classification.csv')
    class_call = (
        f'--problem classification --train {class_train_path} --test {class_train_path} --target Y '
        '--preset fast_train --timeout 0.1 --depth 3 --arity 3 '
        '--popsize 3 --gen_num 5 --tuning 1'
    ).split()

    ts_predictions = call_cli_with_parameters(ts_call)
    assert ts_predictions is not None
    class_predictions = call_cli_with_parameters(class_call)
    assert class_predictions is not None
