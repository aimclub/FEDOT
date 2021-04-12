import os
import numpy as np

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from examples.regression_with_tuning_example import get_regression_dataset


def get_chain():
    node_scaling = PrimaryNode('scaling')
    node_ransac = SecondaryNode('ransac_lin_reg', nodes_from=[node_scaling])
    node_ridge = SecondaryNode('lasso', nodes_from=[node_ransac])
    chain = Chain(node_ridge)

    return chain


def create_correct_path(path: str, dirname_flag: bool = False):
    """
    Create path with time which was created during the testing process.
    """

    for dirname in next(os.walk(os.path.curdir))[1]:
        if dirname.endswith(path):
            if dirname_flag:
                return dirname
            else:
                file = os.path.join(dirname, path + '.json')
                return file
    return None


def run_import_export_example(chain_path):
    features_options = {'informative': 1, 'bias': 0.0}
    samples_amount = 100
    features_amount = 2
    x_train, y_train, x_test, y_test = get_regression_dataset(features_options,
                                                              samples_amount,
                                                              features_amount)

    # Define regression task
    task = Task(TaskTypesEnum.regression)

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(x_train)),
                            features=x_train,
                            target=y_train,
                            task=task,
                            data_type=DataTypesEnum.table)

    predict_input = InputData(idx=np.arange(0, len(x_test)),
                              features=x_test,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.table)

    # Get chain and fit it
    chain = get_chain()
    chain.fit_from_scratch(train_input)

    predicted_output = chain.predict(predict_input)
    prediction_before_export = np.array(predicted_output.predict)
    print(f'Before export {prediction_before_export[:4]}')

    # Export it
    chain.save(path=chain_path)

    # Import chain
    json_path_load = create_correct_path(chain_path)
    new_chain = Chain()
    new_chain.load(json_path_load)

    predicted_output_after_export = new_chain.predict(predict_input)
    prediction_after_export = np.array(predicted_output_after_export.predict)

    print(f'After import {prediction_after_export[:4]}')


if __name__ == '__main__':
    run_import_export_example(chain_path='import_export')
