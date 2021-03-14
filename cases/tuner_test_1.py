from pylab import rcParams
rcParams['figure.figsize'] = 18, 7
import warnings
warnings.filterwarnings('ignore')
import datetime

from cases.tuner_test_supplementary import *


def tuner_function_20(chain, train_input):
    chain.fine_tune_all_nodes(train_input,
                              max_lead_time=datetime.timedelta(minutes=2),
                              iterations=30)

    # Fit it
    chain.fit_from_scratch(train_input)

    return chain


def tuner_function_100(chain, train_input):
    chain.fine_tune_all_nodes(train_input,
                              max_lead_time=datetime.timedelta(minutes=5),
                              iterations=100)

    # Fit it
    chain.fit_from_scratch(train_input)

    return chain


def run_experiment(tuner_iterations_function, folder_to_save):
    all_iterations = 100

    ######################
    #  Regression cases  #
    ######################
    for case, case_name in zip([run_rivers_case_regression,
                                run_synthetic_case_regression],
                               ['Real_case_reg.csv',
                                'Synthetic_case_reg.csv']):
        print(f'Processing case for file {case_name}...')

        first_reg_chain = reg_chain_1()
        second_reg_chain = reg_chain_2()
        third_reg_chain = reg_chain_3()
        for j, chain_struct in enumerate([first_reg_chain, second_reg_chain, third_reg_chain]):
            result_df = case(chain=chain_struct,
                             iterations=all_iterations,
                             tuner_function=tuner_iterations_function)

            if j == 0:
                case_reg_report = result_df
            else:
                frames = [case_reg_report, result_df]
                case_reg_report = pd.concat(frames)

        # Save to file
        case_reg_file = os.path.join(folder_to_save, case_name)
        case_reg_report.to_csv(case_reg_file, index=False)

    ##########################
    #  Classification cases  #
    ##########################
    case_name = 'Synthetic_case_class.csv'
    print(f'Processing case for file {case_name}...')

    first_class_chain = class_chain_1()
    second_class_chain = class_chain_2()
    third_class_chain = class_chain_3()
    for j, chain_struct in enumerate([first_class_chain, second_class_chain, third_class_chain]):
        result_df = run_synthetic_case_classification(chain=chain_struct,
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
                   folder_to_save='D:/ITMO/tuning_exp/1_fine_tune_all_nodes/20')
