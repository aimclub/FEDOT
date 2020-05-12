import csv
import datetime
import gc
import os

from benchmark.experimens.credit_scoring_experiment import run_credit_scoring_problem
from core.composer.optimisers.crossover import CrossoverTypesEnum
from core.composer.optimisers.gp_optimiser import GPChainOptimiserParameters
from core.composer.optimisers.mutation import MutationTypesEnum
from core.composer.optimisers.regularization import RegularizationTypesEnum
from core.composer.optimisers.selection import SelectionTypesEnum
from core.utils import project_root


def write_header_to_csv(f):
    f = f'../../../tmp/{f}'
    if not os.path.isdir('../../../tmp'):
        os.mkdir('../../../tmp')
    with open(f, 'w', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        writer.writerow(['t_opt', 'regular', 'AUC', 'n_models', 'n_layers'])


def add_result_to_csv(f, t_opt, regular, auc, n_models, n_layers):
    f = f'../../../tmp/{f}'
    with open(f, 'a', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        writer.writerow([t_opt, regular, auc, n_models, n_layers])


if __name__ == '__main__':
    max_amount_of_time = 60
    step = 10
    file_path_train = 'cases/data/scoring/scoring_train.csv'
    full_path_train = os.path.join(str(project_root()), file_path_train)
    file_path_test = 'cases/data/scoring/scoring_test.csv'
    full_path_test = os.path.join(str(project_root()), file_path_test)
    file_path_result = 'regular_exp.csv'

    write_header_to_csv(file_path_result)

    time_amount = step
    while time_amount < max_amount_of_time:
        for regular_type in (RegularizationTypesEnum.none, RegularizationTypesEnum.decremental):
            gc.collect()

            selection_types = [SelectionTypesEnum.tournament]
            crossover_types = [CrossoverTypesEnum.subtree]
            mutation_types = [MutationTypesEnum.simple, MutationTypesEnum.growth, MutationTypesEnum.reduce]
            optimiser_parameters = GPChainOptimiserParameters(selection_types=selection_types,
                                                              crossover_types=crossover_types,
                                                              mutation_types=mutation_types,
                                                              regularization_type=regular_type)
            roc_auc, chain = run_credit_scoring_problem(full_path_train, full_path_test,
                                                        max_lead_time=datetime.timedelta(minutes=time_amount),
                                                        gp_optimiser_params=optimiser_parameters)

            is_regular = True if regular_type == RegularizationTypesEnum.decremental else False
            add_result_to_csv(file_path_result, time_amount, is_regular, round(roc_auc, 4), len(chain.nodes),
                              chain.depth)

        time_amount += step
