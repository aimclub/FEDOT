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
                              max_lead_time=datetime.timedelta(minutes=50),
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
                              max_lead_time=datetime.timedelta(minutes=50),
                              iterations=iterations_per_node)

    # Fit it
    chain.fit_from_scratch(train_input)

    return chain


def run_classification_experiment(tuner_iterations_function,
                                  folder_to_save,
                                  dataset_number):
    create_folder(os.path.abspath(folder_to_save))
    name_class_by_number = {1: 'class_Amazon_employee_access.csv',
                            2: 'class_cnae-9.csv',
                            3: 'class_volkert_small.csv'}
    run_class_by_number = {1: run_class_amazon_employee_access,
                           2: run_class_cnae9,
                           3: run_class_volkert_small}
    # Amount of launches
    all_iterations = 2

    #########################
    #  Classification case  #
    #########################
    case_name = name_class_by_number.get(dataset_number)
    print(f'Processing case for file {case_name}...')

    first_class_chain = class_chain_1()
    second_class_chain = class_chain_2()
    third_class_chain = class_chain_3()
    for j, chain_struct in enumerate([first_class_chain, second_class_chain, third_class_chain]):
        launch = run_class_by_number.get(dataset_number)
        result_df = launch(chain=chain_struct,
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


def run_regression_experiment(tuner_iterations_function, folder_to_save,
                              dataset_number):
    create_folder(os.path.abspath(folder_to_save))
    name_reg_by_number = {1: 'reg_cal_housing.csv',
                          2: 'reg_delta_ailerons.csv',
                          3: 'reg_pol.csv'}
    run_reg_by_number = {1: run_reg_cal_housing,
                         2: run_reg_delta_ailerons,
                         3: run_reg_pol}
    # Amount of launches
    all_iterations = 5

    #####################
    #  Regression case  #
    #####################
    case_name = name_reg_by_number.get(dataset_number)
    print(f'Processing case for file {case_name}...')

    first_reg_chain = reg_chain_1()
    second_reg_chain = reg_chain_2()
    third_reg_chain = reg_chain_3()
    for j, chain_struct in enumerate([first_reg_chain, second_reg_chain, third_reg_chain]):
        launch = run_reg_by_number.get(dataset_number)
        result_df = launch(chain=chain_struct,
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


if __name__ == '__main__':
    ####################################################
    #       Old hp_tuning - Serial isolated tuning     #
    ####################################################

    # 3 case for every task
    for dataset_number in [1, 2, 3]:
        print(f'DATASET NUMBER {dataset_number}')
        # Run old tuner with 20 iterations for regression task
        run_regression_experiment(tuner_iterations_function=tuner_function_20,
                                  folder_to_save='isolated_tuner/20',
                                  dataset_number=dataset_number)

        # Run old tuner with 100 iterations for regression task
        run_regression_experiment(tuner_iterations_function=tuner_function_100,
                                  folder_to_save='isolated_tuner/100',
                                  dataset_number=dataset_number)

        # Run old tuner with 20 iterations for classification task
        run_classification_experiment(tuner_iterations_function=tuner_function_20,
                                      folder_to_save='isolated_tuner/20',
                                      dataset_number=dataset_number)

        # Run old tuner with 100 iterations for classification task
        run_classification_experiment(tuner_iterations_function=tuner_function_100,
                                      folder_to_save='isolated_tuner/100',
                                      dataset_number=dataset_number)
