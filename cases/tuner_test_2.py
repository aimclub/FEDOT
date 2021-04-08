from pylab import rcParams
rcParams['figure.figsize'] = 18, 7
import warnings
warnings.filterwarnings('ignore')
import datetime

from cases.tuner_test_supplementary import *
from fedot.core.models.tuning.hyperopt_tune.tuners import ChainTuner
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score as roc_auc
from sklearn.metrics import make_scorer


def tuner_function_20_reg(chain, train_input):
    chain_tuner = ChainTuner(chain=chain,
                             task=train_input.task,
                             iterations=20)
    tuned_chain = chain_tuner.tune_chain(input_data=train_input,
                                         loss_function=mean_absolute_error)

    return tuned_chain


def tuner_function_100_reg(chain, train_input):
    chain_tuner = ChainTuner(chain=chain,
                             task=train_input.task,
                             iterations=100)
    tuned_chain = chain_tuner.tune_chain(input_data=train_input,
                                         loss_function=mean_absolute_error)

    return tuned_chain


def tuner_function_20_class(chain, train_input):
    chain_tuner = ChainTuner(chain=chain,
                             task=train_input.task,
                             iterations=20)
    tuned_chain = chain_tuner.tune_chain(input_data=train_input,
                                         loss_function=roc_auc)

    return tuned_chain


def tuner_function_100_class(chain, train_input):
    chain_tuner = ChainTuner(chain=chain,
                             task=train_input.task,
                             iterations=100)
    tuned_chain = chain_tuner.tune_chain(input_data=train_input,
                                         loss_function=roc_auc)

    return tuned_chain


def run_experiment(tuner_iterations, folder_to_save):
    all_iterations = 100
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
    # case_name = 'Pnn_regression.csv'
    # print(f'Processing case for file {case_name}...')
    #
    # first_reg_chain = reg_chain_1()
    # second_reg_chain = reg_chain_2()
    # third_reg_chain = reg_chain_3()
    # for j, chain_struct in enumerate([first_reg_chain, second_reg_chain, third_reg_chain]):
    #     result_df = run_pnn_regression(chain=chain_struct,
    #                                    iterations=all_iterations,
    #                                    tuner_function=tuner_iterations_function_reg)
    #
    #     if j == 0:
    #         case_reg_report = result_df
    #     else:
    #         frames = [case_reg_report, result_df]
    #         case_reg_report = pd.concat(frames)
    #
    # # Save to file
    # case_reg_file = os.path.join(folder_to_save, case_name)
    # case_reg_report.to_csv(case_reg_file, index=False)

    #########################
    #  Classification case  #
    #########################
    case_name = 'Pnn_classification.csv'
    print(f'Processing case for file {case_name}...')

    first_class_chain = class_chain_1()
    second_class_chain = class_chain_2()
    third_class_chain = class_chain_3()
    for j, chain_struct in enumerate([first_class_chain, second_class_chain, third_class_chain]):
        result_df = run_pnn_1_classification(chain=chain_struct,
                                             iterations=all_iterations,
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
    #              New tuning - ChainTuner             #
    ####################################################
    run_experiment(tuner_iterations=20,
                   folder_to_save='D:/ITMO/tuning_exp_2/chain_tuner/20')

    run_experiment(tuner_iterations=100,
                   folder_to_save='D:/ITMO/tuning_exp_2/chain_tuner/100')
