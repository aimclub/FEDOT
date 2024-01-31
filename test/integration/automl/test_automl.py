from examples.advanced.automl.h2o_example import h2o_classification_pipeline_evaluation, \
    h2o_regression_pipeline_evaluation, h2o_ts_pipeline_evaluation
from fedot.core.repository.operation_types_repository import OperationTypesRepository


def test_h2o_vs_fedot_example():
    h2o_classification_pipeline_evaluation()
    h2o_regression_pipeline_evaluation()
    h2o_ts_pipeline_evaluation()
