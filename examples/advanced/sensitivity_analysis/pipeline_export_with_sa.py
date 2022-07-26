import os

import numpy as np

from examples.advanced.sensitivity_analysis.dataset_access import get_scoring_data
from examples.advanced.sensitivity_analysis.pipelines_access import get_three_depth_manual_class_pipeline
from fedot.core.pipelines.pipeline import Pipeline
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


def run_import_export_example(pipeline_path):
    # Prepare data to train the model
    train_data, test_data = get_scoring_data()

    # Get pipeline and fit it
    pipeline = get_three_depth_manual_class_pipeline()
    pipeline.fit_from_scratch(train_data)

    predicted_output = pipeline.predict(test_data)
    prediction_before_export = np.array(predicted_output.predict)
    print(f'Before export {prediction_before_export[:4]}')

    NodesAnalysis(pipeline, train_data, test_data,
                  approaches=[NodeDeletionAnalyze,
                              NodeReplaceOperationAnalyze]).analyze()

    # Export it
    pipeline.save(path=pipeline_path)

    # Import pipeline
    json_path_load = create_correct_path(pipeline_path)
    new_pipeline = Pipeline.from_serialized(json_path_load)

    predicted_output_after_export = new_pipeline.predict(test_data)
    prediction_after_export = np.array(predicted_output_after_export.predict)

    print(f'After import {prediction_after_export[:4]}')


if __name__ == '__main__':
    run_import_export_example(pipeline_path='import_export_sa')
