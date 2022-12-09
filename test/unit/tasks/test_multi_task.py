import numpy as np

from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import TaskTypesEnum

from examples.advanced.multitask_classification_regression import prepare_multitask_data


def test_multitask_pipeline_predict_correctly():
    """ Test pipeline fit predict correctness for classification and regression task in one pipeline """
    train_multimodal, test_multimodal = prepare_multitask_data()

    logit_node = PrimaryNode('logit')
    data_source_node = PrimaryNode('data_source_table/regression')
    final_node = SecondaryNode('rfr', nodes_from=[logit_node, data_source_node])
    final_node.parameters = {'n_estimators': 5}
    multitask_pipeline = Pipeline(final_node)

    multitask_pipeline.fit(train_multimodal)
    side_pipeline = multitask_pipeline.pipeline_for_side_task(task_type=TaskTypesEnum.classification)

    # Replace the name of main "data source" in preprocessor
    side_pipeline.preprocessor.main_target_source_name = 'logit'
    side_predict = np.ravel(side_pipeline.predict(test_multimodal, output_mode='labels').predict)
    main_predict = multitask_pipeline.predict(test_multimodal).predict

    assert np.array_equal(side_predict, np.array(['a_category', 'a_category', 'b_category', 'b_category']))
    # Two source features and predicted class label as third
    assert multitask_pipeline.root_node.fitted_operation.n_features_in_ == 3
