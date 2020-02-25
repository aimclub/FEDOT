from core.composer.gp_composer.gp_composer import GPComposer,GPComposer_requirements
from core.models.model import XGBoost, LogRegression, KNN

def test_composer():
    composer = GPComposer()
    composer_requirements = GPComposer_requirements(primary_requirements=[LogRegression(), KNN()],
                                       secondary_requirements=[LogRegression(), XGBoost()], max_arity=3, max_depth=5,pop_size=10)
    new_chain = composer.compose_chain(data=None,initial_chain=None,
                                       composer_requirements= composer_requirements,
                                       metrics=None)


if __name__ == '__main__':
    test_composer()