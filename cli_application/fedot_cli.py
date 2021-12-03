import argparse

parser = argparse.ArgumentParser(description="An argparse example")

parser.add_argument('--problem',
                    help='The name of modelling problem to solve: \n'
                    'classification;'
                    'regression;'
                    'ts_forecasting;'
                    'clustering',
                    required=True)

parser.add_argument('--preset',
                    help='Name of preset for model building: \n'
                    'light;'
                    'light_tun;'
                    'light_steady_state;'
                    'ultra_light;'
                    'ultra_light_tun;'
                    'ultra_steady_state;'
                    'ts;'
                    'ts_tun;'
                    'gpu')

parser.add_argument('--timeout',
                    help='Time for model design (in minutes)')
parser.add_argument('--seed',
                    help='Value for fixed random seed')

parser.add_argument('--train',
                    help='Path to train data file')
parser.add_argument('--test',
                    help='Path to test data file')

# composer parameters
parser.add_argument('--c_depth',
                    help='Composer parameter: max depth of the pipeline')
parser.add_argument('--c_arity',
                    help='Composer parameter: max arity of the pipeline nodes')
parser.add_argument('--c_popsize',
                    help='Composer parameter: population size')
parser.add_argument('--c_gen_num',
                    help='Composer parameter: number of generations')
parser.add_argument('--c_timeout',
                    help='Composer parameter: composing time (minutes)')
parser.add_argument('--c_opers',
                    help='Composer parameter: model names to use',
                    nargs='*')
parser.add_argument('--c_tuning',
                    help='Composer parameter: 1 - with tuning, 0 - without tuning')
parser.add_argument('--c_cv_folds',
                    help='Composer parameter: Number of folds for cross-validation')
parser.add_argument('--c_val_bl',
                    help='Composer parameter: Number of validation blocks for time series forecasting')
parser.add_argument('--c_hist_path',
                    help='Composer parameter: Name of the folder for composing history')


args = parser.parse_args()




