from datetime import timedelta

from sklearn.metrics import roc_auc_score as roc_auc

from examples.real_cases.data.data_utils import get_scoring_case_data_paths
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.operation_types_repository import OperationTypesRepository


# TODO not working now - add switch to other repository.json
def run_pipeline_from_automl(train_file_path: str, test_file_path: str,
                             max_run_time: timedelta = timedelta(minutes=10)):
    """ Function run pipeline with Auto ML models in nodes

    :param train_file_path: path to the csv file with data for train
    :param test_file_path: path to the csv file with data for validation
    :param max_run_time: maximum running time for customization of the "tpot" model

    :return roc_auc_value: ROC AUC metric for pipeline
    """
    with OperationTypesRepository.init_automl_repository() as _:
        train_data = InputData.from_csv(train_file_path)
        test_data = InputData.from_csv(test_file_path)

        testing_target = test_data.target

        node_scaling = PipelineNode('scaling')
        node_tpot = PipelineNode('tpot_class')

        node_tpot.parameters = {'timeout': max_run_time.seconds}

        node_lda = PipelineNode('lda', nodes_from=[node_scaling])
        node_rf = PipelineNode('rf', nodes_from=[node_tpot, node_lda])
        pipeline = Pipeline(node_rf)

        pipeline.fit(train_data)
        results = pipeline.predict(test_data)

        roc_auc_value = roc_auc(y_true=testing_target,
                                y_score=results.predict)
        print(roc_auc_value)
    return roc_auc_value


if __name__ == '__main__':
    train_file_path, test_file_path = get_scoring_case_data_paths()
    run_pipeline_from_automl(train_file_path, test_file_path)
