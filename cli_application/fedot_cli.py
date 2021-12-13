import argparse
from argparse import RawTextHelpFormatter
from fedot.core.repository.tasks import TsForecastingParams
from fedot.api.main import Fedot
import os


def add_args_to_parser(parser, tag, help, required=False, is_list=False):
    """ Function for formalized parser arguments creation
    :param parser: ArgumentParser class object
    :param tag: str, name of parameter to parse
    :param help: str, help text for parameter
    :param required: bool, is the parameter required or not
    :param is_list: bool, does the parameter contain multiple input
    """
    if is_list is True:
        nargs = '*'
    elif is_list is False:
        nargs = None
    parser.add_argument(tag, help=help, required=required, nargs=nargs)


def remove_none_keys(parameters):
    """ Function that removes nan parameters from input
    and converts numbers from strings
    :param parameters: dict with argparse filtered keys
    """
    for k, v in list(parameters.items()):
        if v is None:
            del parameters[k]
        elif type(v) is not bool:
            try:
                parameters[k] = float(v)
            except Exception:
                pass

# parameters to init Fedot class
main_params_names = ['problem', 'preset', 'timeout', 'seed']
# parameters to fit model
fit_params_names = ['train', 'target']
# composer parameters
composer_params_names = ['depth', 'arity', 'popsize', 'gen_num', 'c_timeout',
                         'opers', 'tuning', 'cv_folds', 'val_bl', 'hist_path']
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
                            'light_tun;\n'
                            'light_steady_state;\n'
                            'ultra_light;\n'
                            'ultra_light_tun;\n'
                            'ultra_steady_state;\n'
                            'ts;\n'
                            'ts_tun;\n'
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
                   {'tag': '--c_timeout',
                    'help': 'Composer parameter: composing time (minutes)'},
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
              'c_timeout': 'timeout',
              'opers': 'available_operations',
              'tuning': 'with_tuning',
              'cv_folds': 'cv_folds',
              'val_bl': 'validation_blocks',
              'hist_path': 'history_folder',
              'for_len': 'forecast_length'
              }

parser = argparse.ArgumentParser(description="__________________ FEDOT API console call __________________",
                                 formatter_class=RawTextHelpFormatter)

for argument_dict in arguments_dicts:
    add_args_to_parser(parser=parser, **argument_dict)
parameters = parser.parse_args()

composer_params = {}
main_params = {}
fit_params = {}
for arg in vars(parameters):
    if arg in composer_params_names:
        composer_params[keys_names[arg]] = getattr(parameters, arg)
    if arg in main_params_names:
        main_params[keys_names[arg]] = getattr(parameters, arg)
    if arg in fit_params_names:
        fit_params[keys_names[arg]] = getattr(parameters, arg)

if main_params['problem'] == 'ts_forecasting' and getattr(parameters, 'for_len') is not None:
    main_params['task_params'] = TsForecastingParams(forecast_length=int(getattr(parameters, 'for_len')))
else:
    raise ValueError("Forecast length (for_len) is necessary parameter for ts_forecasting problem")

if composer_params['with_tuning'] == '1':
    composer_params['with_tuning'] = True
else:
    composer_params['with_tuning'] = False

remove_none_keys(main_params)
remove_none_keys(composer_params)
remove_none_keys(fit_params)

main_params['composer_params'] = composer_params

model = Fedot(**main_params)
print("\nFitting start...")
model.fit(**fit_params)
print("\nPrediction start...")
prediction = model.predict(features=getattr(parameters, 'test'), save_predictions=True)
print(f"\nPrediction saved at {os.getcwd()}\\predictions.csv")
