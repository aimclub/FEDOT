from pylab import rcParams
rcParams['figure.figsize'] = 18, 7
import warnings
warnings.filterwarnings('ignore')
import datetime

from cases.tuner_test_supplementary import *


def tuner_function_20(chain, train_input):
    # Calculate amount of iterations we can apply per node
    nodes_amount = len(chain.nodes)
    iterations_per_node = round(20 / nodes_amount)
    iterations_per_node = int(iterations_per_node)

    chain.fine_tune_all_nodes(train_input,
                              max_lead_time=datetime.timedelta(minutes=2),
                              iterations=iterations_per_node)

    # Fit it
    chain.fit_from_scratch(train_input)

    return chain


def tuner_function_100(chain, train_input):
    # Calculate amount of iterations we can apply per node
    nodes_amount = len(chain.nodes)
    iterations_per_node = round(100 / nodes_amount)
    iterations_per_node = int(iterations_per_node)

    chain.fine_tune_all_nodes(train_input,
                              max_lead_time=datetime.timedelta(minutes=5),
                              iterations=iterations_per_node)

    # Fit it
    chain.fit_from_scratch(train_input)

    return chain


def run_experiment(tuner_iterations_function, folder_to_save):
    all_iterations = 100

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
    #                                    tuner_function=tuner_iterations_function)
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
        result_df = run_pnn_classification(chain=chain_struct,
                                           iterations=all_iterations,
                                           tuner_function=tuner_iterations_function)

        if j == 0:
            case_class_report = result_df
        else:
            frames = [case_class_report, result_df]
            case_class_report = pd.concat(frames)

    # Save to file
    case_class_file = os.path.join(folder_to_save, case_name)
    case_class_report.to_csv(case_class_file, index=False)


if __name__ == '__main__':
    run_experiment(tuner_iterations_function=tuner_function_20,
                   folder_to_save='D:/ITMO/tuning_exp_2/old_tuner/20')

    run_experiment(tuner_iterations_function=tuner_function_100,
                   folder_to_save='D:/ITMO/tuning_exp_2/old_tuner/100')
