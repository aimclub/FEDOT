import os

from fedot.api.fedot_cli import create_parser, separate_argparse_to_fedot, preprocess_keys, run_fedot, \
    arguments_dicts
from fedot.core.utils import fedot_project_root

project_root_path = str(fedot_project_root())
ts_train_path = os.path.join(project_root_path, 'test/data/simple_time_series.csv')
ts_call = f'--problem ts_forecasting --preset fast_train --timeout 0.1 --depth 3 --arity 3 \
                                    --popsize 3 --gen_num 5 --opers lagged linear ridge --tuning \
                                    0 --cv_folds 2 --val_bl 2 --target sea_height --train {ts_train_path} \
                                    --test {ts_train_path} --for_len 10'.split()

class_train_path = os.path.join(project_root_path, 'test/data/simple_classification.csv')
class_call = f'--problem classification --train {class_train_path} --test {class_train_path} --target Y \
                                     --preset fast_train --timeout 0.1 --depth 3 --arity 3 \
                                     --popsize 3 --gen_num 5 --tuning 1'.split()


def call_cli_with_parameters(call_string):
    """ Function that imitates argparse api call"""
    parser = create_parser(arguments_dicts)
    parameters = parser.parse_args(call_string)
    main_params, fit_params = separate_argparse_to_fedot(parameters)
    preprocess_keys(main_params)
    preprocess_keys(fit_params)
    predictions = run_fedot(parameters, main_params, fit_params, save_predictions=False)
    return predictions


def test_cli_with_parameters():
    """ Test all parameters used in cli are available from api"""
    ts_predictions = call_cli_with_parameters(ts_call)
    assert ts_predictions is not None
    class_predictions = call_cli_with_parameters(class_call)
    assert class_predictions is not None
