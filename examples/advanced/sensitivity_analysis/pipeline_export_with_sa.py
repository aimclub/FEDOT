import numpy as np

from examples.advanced.sensitivity_analysis.dataset_access import get_scoring_data
from examples.advanced.sensitivity_analysis.pipelines_access import get_three_depth_manual_class_pipeline
from fedot.core.pipelines.pipeline import Pipeline
from fedot.sensitivity.node_sa_approaches import NodeDeletionAnalyze, NodeReplaceOperationAnalyze
from fedot.sensitivity.nodes_sensitivity import NodesAnalysis


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
    pipeline.save(path=pipeline_path, create_subdir=False)

    # Import pipeline
    new_pipeline = Pipeline().load(pipeline_path)

    predicted_output_after_export = new_pipeline.predict(test_data)
    prediction_after_export = np.array(predicted_output_after_export.predict)

    print(f'After import {prediction_after_export[:4]}')


if __name__ == '__main__':
    run_import_export_example(pipeline_path='import_export_sa')
