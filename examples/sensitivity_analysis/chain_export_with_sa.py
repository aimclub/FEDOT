import os

import numpy as np

from examples.sensitivity_analysis.chains_access import get_three_depth_manual_class_chain
from examples.sensitivity_analysis.dataset_access import get_scoring_data
from fedot.core.chains.chain import Chain
from fedot.sensitivity.node_sa_approaches import NodeDeletionAnalyze, NodeReplaceOperationAnalyze
from fedot.sensitivity.nodes_sensitivity import NodesAnalysis


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
    # Prepare data to train the model
    train_data, test_data = get_scoring_data()

    # Get chain and fit it
    chain = get_three_depth_manual_class_chain()
    chain.fit_from_scratch(train_data)

    predicted_output = chain.predict(test_data)
    prediction_before_export = np.array(predicted_output.predict)
    print(f'Before export {prediction_before_export[:4]}')

    NodesAnalysis(chain, train_data, test_data,
                  approaches=[NodeDeletionAnalyze,
                              NodeReplaceOperationAnalyze]).analyze()

    # Export it
    chain.save(path=chain_path)

    # Import chain
    json_path_load = create_correct_path(chain_path)
    new_chain = Chain()
    new_chain.load(json_path_load)

    predicted_output_after_export = new_chain.predict(test_data)
    prediction_after_export = np.array(predicted_output_after_export.predict)

    print(f'After import {prediction_after_export[:4]}')


if __name__ == '__main__':
    run_import_export_example(chain_path='import_export_sa')
