import argparse
from argparse import RawTextHelpFormatter
from fedot.core.repository.tasks import TsForecastingParams
from fedot.api.main import Fedot
import os


def create_parser(arguments_list):
    """ Function for creating parser object and adding arguments"""
    parser = argparse.ArgumentParser(description="__________________ FEDOT API console call __________________",
                                     formatter_class=RawTextHelpFormatter)

    for argument_dict in arguments_list:
        add_args_to_parser(parser=parser, **argument_dict)
    return parser


def add_args_to_parser(parser, tag: str, help: str, required: bool = False, is_list: bool = False):
    """ Function for formalized parser arguments creation
    :param parser: ArgumentParser class object
    :param tag: name of parameter to parse
    :param help: help text for parameter
    :param required: is the parameter required or not
    :param is_list: does the parameter contain multiple input
    """
    if is_list is True:
        nargs = '*'
    else:
        nargs = None
    parser.add_argument(tag, help=help, required=required, nargs=nargs)


def separate_argparse_to_fedot(parameters):
    """ Function for separating argparse parameters on fedot fit/predict/composer"""
    main_params = {}
    fit_params = {}
    for arg in vars(parameters):
        if arg in main_params_names:
            main_params[keys_names[arg]] = getattr(parameters, arg)
        if arg in fit_params_names:
            fit_params[keys_names[arg]] = getattr(parameters, arg)

    if main_params['problem'] == 'ts_forecasting' and getattr(parameters, 'for_len') is not None:
        main_params['task_params'] = TsForecastingParams(forecast_length=int(getattr(parameters, 'for_len')))
    elif main_params['problem'] == 'ts_forecasting' and getattr(parameters, 'for_len') is None:
        raise ValueError("Forecast length (for_len) is necessary parameter for ts_forecasting problem")

    if main_params['with_tuning'] == '1':
        main_params['with_tuning'] = True
    else:
        main_params['with_tuning'] = False
    return main_params, fit_params


def preprocess_keys(parameters: dict):
    """ Function that removes nan parameters from input
    and converts numbers from strings
    :param parameters: argparse filtered keys
    """
    for k, v in list(parameters.items()):
        if v is None:
            del parameters[k]
        elif type(v) is not bool:
            try:
                parameters[k] = float(v)
            except Exception:
                pass


def run_fedot(parameters, main_params, fit_params, save_predictions=True):
    """ Function for run prediction on fedot """
    model = Fedot(**main_params)
    print("\nFitting start...")
    model.fit(**fit_params)
    print("\nPrediction start...")
    prediction = model.predict(features=getattr(parameters, 'test'), in_sample=False, save_predictions=save_predictions)
    print(f"\nPrediction saved at {os.getcwd()}\\predictions.csv")
    return prediction


# parameters to init Fedot class
main_params_names = ['problem', 'timeout', 'seed', 'depth', 'arity', 'popsize', 'gen_num',
                     'opers', 'tuning', 'cv_folds', 'val_bl', 'hist_path', 'preset']
# parameters to fit model
fit_params_names = ['train', 'target']

# dictionary with keys for parser creation
arguments_dicts = [{'tag': '--problem',
                    'help': 'The name of modelling problem to solve: \n'
                            'classification;\n'
                            'regression;\n'
                            'ts_forecasting;\n'
                            'clustering',
                    'required': True},
                   {'tag': '--train',
                    'help': 'Path to train data file',
                    'required': True},
                   {'tag': '--test',
                    'help': 'Path to test data file',
                    'required': True},
                   {'tag': '--preset',
                    'help': 'Name of preset for model building: \n'
                            'light;\n'
                            'light_steady_state;\n'
                            'ultra_light;\n'
                            'ultra_steady_state;\n'
                            'ts;\n'
                            'gpu'},
                   {'tag': '--timeout',
                    'help': 'Time for model design (in minutes)'},
                   {'tag': '--seed',
                    'help': 'Value for fixed random seed'},
                   {'tag': '--target',
                    'help': 'Name of target variable in data'},
                   {'tag': '--depth',
                    'help': 'Composer parameter: max depth of the pipeline'},
                   {'tag': '--arity',
                    'help': 'Composer parameter: max arity of the pipeline nodes'},
                   {'tag': '--popsize',
                    'help': 'Composer parameter: population size'},
                   {'tag': '--gen_num',
                    'help': 'Composer parameter: number of generations'},
                   {'tag': '--opers',
                    'help': 'Composer parameter: model names to use',
                    'is_list': True},
                   {'tag': '--tuning',
                    'help': 'Composer parameter: 1 - with tuning, 0 - without tuning'},
                   {'tag': '--cv_folds',
                    'help': 'Composer parameter: Number of folds for cross-validation'},
                   {'tag': '--val_bl',
                    'help': 'Composer parameter: Number of validation blocks for time series forecasting'},
                   {'tag': '--hist_path',
                    'help': 'Composer parameter: Name of the folder for composing history'},
                   {'tag': '--for_len',
                    'help': 'Time Series Forecasting parameter: forecast length'},
                   ]
# dictionary with tag - name relation
keys_names = {'problem': 'problem',
              'preset': 'preset',
              'timeout': 'timeout',
              'seed': 'seed',
              'train': 'features',
              'target': 'target',
              'depth': 'max_depth',
              'arity': 'max_arity',
              'popsize': 'pop_size',
              'gen_num': 'num_of_generations',
              'opers': 'available_operations',
              'tuning': 'with_tuning',
              'cv_folds': 'cv_folds',
              'val_bl': 'validation_blocks',
              'hist_path': 'history_folder',
              'for_len': 'forecast_length'
              }


def run_cli():
    parser = create_parser(arguments_dicts)
    parameters = parser.parse_args()

    main_params, fit_params = separate_argparse_to_fedot(parameters)

    preprocess_keys(main_params)
    preprocess_keys(fit_params)

    run_fedot(parameters, main_params, fit_params)


if __name__ == '__main__':
    run_cli()
