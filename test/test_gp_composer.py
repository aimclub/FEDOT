from core.composer.gp_composer.gp_composer import GPComposer, GPComposerRequirements
from core.models.data import Data
from core.models.model import XGBoost, LogRegression, KNN


def test_composer():
    data = Data.from_csv(
        file_path=f'C://Users//YanaPolonskaya//PycharmProjects//THEODOR(improved-chain)//test//data//agg_data_meta.csv',
        delimiter=';', normalization=True, label='def90')
    # data = Data.from_csv1(
    #    file_path=f'C://Users//YanaPolonskaya//PycharmProjects//THEODOR(improved-chain)//test//data//test_dataset.csv')
    composer = GPComposer()
    composer_requirements = GPComposerRequirements(primary=[LogRegression(), KNN()],
                                                   secondary=[LogRegression(), XGBoost()], max_arity=3,
                                                   max_depth=5, pop_size=10, num_of_generations=10)
    new_chain = composer.compose_chain(data=data, initial_chain=None,
                                       composer_requirements=composer_requirements,
                                       metrics=None)


if __name__ == '__main__':
    test_composer()
