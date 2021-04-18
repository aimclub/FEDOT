import numpy as np

from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.utilities.synthetic.data import regression_dataset

np.random.seed(2020)


def run_experiment():
    features_options = {'informative': 1, 'bias': 0.0}
    samples_amount = 100
    features_amount = 3
    x_data, y_data = regression_dataset(samples_amount=samples_amount,
                                        features_amount=features_amount,
                                        features_options=features_options,
                                        n_targets=1,
                                        noise=0.0, shuffle=True)

    # Define regression task
    task = Task(TaskTypesEnum.regression)

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(x_data)),
                            features=x_data,
                            target=y_data,
                            task=task,
                            data_type=DataTypesEnum.table)

    # Prepare chain
    node_scaling = PrimaryNode('scaling')
    node_final = SecondaryNode('knnreg', nodes_from=[node_scaling])
    node_final.custom_params = {'n_neighbors': 90}
    chain = Chain(node_final)

    # Fit it
    chain.fit(train_input)

    print(chain.predict(train_input).predict)


if __name__ == '__main__':
    # TODO remove file with this example
    run_experiment()
