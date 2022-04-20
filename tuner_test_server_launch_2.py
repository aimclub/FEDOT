from pylab import rcParams
rcParams['figure.figsize'] = 18, 7
import warnings
warnings.filterwarnings('ignore')
import datetime
from datetime import timedelta

from cases.tuner_test_supplementary import *
from fedot.core.models.tuning.hp_tuning.unified import ChainTuner
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score as roc_auc

ALL_ITERATIONS = 30


def tuner_function_20_reg(chain, train_input):
    chain_tuner = ChainTuner(chain=chain,
                             task=train_input.task,
                             iterations=20,
                             max_lead_time=timedelta(minutes=50))
    tuned_chain = chain_tuner.tune_chain(input_data=train_input,
                                         loss_function=mean_absolute_error)

    return tuned_chain


def tuner_function_100_reg(chain, train_input):
    chain_tuner = ChainTuner(chain=chain,
                             task=train_input.task,
                             iterations=100,
                             max_lead_time=timedelta(minutes=50))
    tuned_chain = chain_tuner.tune_chain(input_data=train_input,
                                         loss_function=mean_absolute_error)

    return tuned_chain


def tuner_function_20_class(chain, train_input):
    chain_tuner = ChainTuner(chain=chain,
                             task=train_input.task,
                             iterations=20,
                             max_lead_time=timedelta(minutes=50))
    tuned_chain = chain_tuner.tune_chain(input_data=train_input,
                                         loss_function=roc_auc,
                                         loss_params={'multi_class': 'ovr'})

    return tuned_chain


def tuner_function_100_class(chain, train_input):
    chain_tuner = ChainTuner(chain=chain,
                             task=train_input.task,
                             iterations=100,
                             max_lead_time=timedelta(minutes=50))
    tuned_chain = chain_tuner.tune_chain(input_data=train_input,
                                         loss_function=roc_auc,
                                         loss_params={'multi_class': 'ovr'})

    return tuned_chain


def run_experiment(tuner_iterations, folder_to_save, dataset_number):
    create_folder(os.path.abspath(folder_to_save))
    if tuner_iterations == 20:
        tuner_iterations_function_reg = tuner_function_20_reg
        tuner_iterations_function_class = tuner_function_20_class
    elif tuner_iterations == 100:
        tuner_iterations_function_reg = tuner_function_100_reg
        tuner_iterations_function_class = tuner_function_100_class
    else:
        raise ValueError('"tuner_iterations" must be equal to 20 or 100')

    #####################
    #  Regression case  #
    #####################
    name_reg_by_number = {1: 'reg_cal_housing.csv',
                          2: 'reg_delta_ailerons.csv',
                          3: 'reg_pol.csv'}
    run_reg_by_number = {1: run_reg_cal_housing,
                         2: run_reg_delta_ailerons,
                         3: run_reg_pol}

    case_name = name_reg_by_number.get(dataset_number)
    print(f'Processing case for file {case_name}...')

    first_reg_chain = reg_chain_1()
    second_reg_chain = reg_chain_2()
    third_reg_chain = reg_chain_3()
    for j, chain_struct in enumerate([first_reg_chain, second_reg_chain, third_reg_chain]):
        launch = run_reg_by_number.get(dataset_number)
        result_df = launch(chain=chain_struct,
                           iterations=ALL_ITERATIONS,
                           tuner_function=tuner_iterations_function_reg)

        if j == 0:
            case_reg_report = result_df
        else:
            frames = [case_reg_report, result_df]
            case_reg_report = pd.concat(frames)

    # Save to file
    case_reg_file = os.path.join(folder_to_save, case_name)
    case_reg_report.to_csv(case_reg_file, index=False)

    #########################
    #  Classification case  #
    #########################
    name_class_by_number = {1: 'class_Amazon_employee_access.csv',
                            2: 'class_cnae-9.csv',
                            3: 'class_volkert_small.csv'}
    run_class_by_number = {1: run_class_amazon_employee_access,
                           2: run_class_cnae9,
                           3: run_class_volkert_small}

    case_name = name_class_by_number.get(dataset_number)
    print(f'Processing case for file {case_name}...')

    first_class_chain = class_chain_1()
    second_class_chain = class_chain_2()
    third_class_chain = class_chain_3()
    for j, chain_struct in enumerate([first_class_chain, second_class_chain, third_class_chain]):
        launch = run_class_by_number.get(dataset_number)
        result_df = launch(chain=chain_struct,
                           iterations=ALL_ITERATIONS,
                           tuner_function=tuner_iterations_function_class)

        if j == 0:
            case_class_report = result_df
        else:
            frames = [case_class_report, result_df]
            case_class_report = pd.concat(frames)

    # Save to file
    case_class_file = os.path.join(folder_to_save, case_name)
    case_class_report.to_csv(case_class_file, index=False)


if __name__ == '__main__':
    ####################################################
    #                 New tuning - ChainTuner          #
    ####################################################

    # 3 case for every task
    for dataset_number in [1, 2, 3]:
        print(f'DATASET NUMBER {dataset_number}')
        run_experiment(tuner_iterations=20,
                       folder_to_save='simultaneous_tuner/20',
                       dataset_number=dataset_number)

        run_experiment(tuner_iterations=100,
                       folder_to_save='simultaneous_tuner/100',
                       dataset_number=dataset_number)
