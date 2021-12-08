import argparse
from argparse import RawTextHelpFormatter
from fedot.core.repository.tasks import TsForecastingParams
from fedot.api.main import Fedot


def add_args_to_parser(parser, tag, help, required=False, is_list=False):
    if is_list is True:
        nargs = '*'
    elif is_list is False:
        nargs = None
    parser.add_argument(tag, help=help, required=required, nargs=nargs)
    return


def remove_none_keys(parameters):
    for k, v in list(parameters.items()):
        #print(k, v)
        if v is None:
            del parameters[k]
        elif type(v) is not bool:
            try:
                parameters[k] = float(v)
            except Exception as e:
                print(e)


arguments_dicts = [{'tag': '--m_problem',
                    'help': 'The name of modelling problem to solve: \n'
                            'classification;\n'
                            'regression;\n'
                            'ts_forecasting;\n'
                            'clustering',
                    'required': True},
                   {'tag': '--f_train',
                    'help': 'Path to train data file',
                    'required': True},
                   {'tag': '--test',
                    'help': 'Path to test data file',
                    'required': True},
                   {'tag': '--m_preset',
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
                   {'tag': '--m_timeout',
                    'help': 'Time for model design (in minutes)'},
                   {'tag': '--m_seed',
                    'help': 'Value for fixed random seed'},
                   {'tag': '--f_target',
                    'help': 'Name of target variable in data'},
                   {'tag': '--c_depth',
                    'help': 'Composer parameter: max depth of the pipeline'},
                   {'tag': '--c_arity',
                    'help': 'Composer parameter: max arity of the pipeline nodes'},
                   {'tag': '--c_popsize',
                    'help': 'Composer parameter: population size'},
                   {'tag': '--c_gen_num',
                    'help': 'Composer parameter: number of generations'},
                   {'tag': '--c_timeout',
                    'help': 'Composer parameter: composing time (minutes)'},
                   {'tag': '--c_opers',
                    'help': 'Composer parameter: model names to use',
                    'is_list': True},
                   {'tag': '--c_tuning',
                    'help': 'Composer parameter: 1 - with tuning, 0 - without tuning'},
                   {'tag': '--c_cv_folds',
                    'help': 'Composer parameter: Number of folds for cross-validation'},
                   {'tag': '--c_val_bl',
                    'help': 'Composer parameter: Number of validation blocks for time series forecasting'},
                   {'tag': '--c_hist_path',
                    'help': 'Composer parameter: Name of the folder for composing history'},
                   {'tag': '--for_len',
                    'help': 'Time Series Forecasting parameter: forecast length'},
                   ]

keys_names = {'m_problem': 'problem',
              'm_preset': 'preset',
              'm_timeout': 'timeout',
              'm_seed': 'seed',
              'f_train': 'features',
              'f_target': 'target',
              'c_depth': 'max_depth',
              'c_arity': 'max_arity',
              'c_popsize': 'pop_size',
              'c_gen_num': 'num_of_generations',
              'c_timeout': 'timeout',
              'c_opers': 'available_operations',
              'c_tuning': 'with_tuning',
              'c_cv_folds': 'cv_folds',
              'c_val_bl': 'validation_blocks',
              'c_hist_path': 'history_folder',
              'for_len': 'forecast_length'
              }

parser = argparse.ArgumentParser(description="An argparse example", formatter_class=RawTextHelpFormatter)
for argument_dict in arguments_dicts:
    add_args_to_parser(parser=parser, **argument_dict)
parameters = parser.parse_args()

composer_params = {}
main_params = {}
fit_params = {}
for arg in vars(parameters):
    if 'c_' in arg:
        composer_params[keys_names[arg]] = getattr(parameters, arg)
    if 'm_' in arg:
        main_params[keys_names[arg]] = getattr(parameters, arg)
    if 'f_' in arg:
        fit_params[keys_names[arg]] = getattr(parameters, arg)

if main_params['problem'] == 'ts_forecasting' and getattr(parameters, 'for_len') is not None:
    main_params['task_params'] = TsForecastingParams(forecast_length=int(getattr(parameters, 'for_len')))

if composer_params['with_tuning'] == '1':
    composer_params['with_tuning'] = True
else:
    composer_params['with_tuning'] = False


remove_none_keys(main_params)
remove_none_keys(composer_params)
remove_none_keys(fit_params)

main_params['composer_params'] = composer_params

print(main_params)
print(fit_params)

model = Fedot(**main_params)
model.fit(**fit_params)
prediction = model.predict(features=getattr(parameters, 'test'))
print(prediction)

