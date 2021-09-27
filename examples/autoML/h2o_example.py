from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from sklearn.metrics import roc_auc_score as roc_auc
from test.unit.tasks.test_classification import get_iris_data


def pipeline_simple() -> Pipeline:
    node = PrimaryNode('h2o')
    pipeline = Pipeline(node)
    return pipeline


def multiclassification_pipeline_fit_correct():
    data = get_iris_data()
    pipeline = pipeline_simple()
    train_data, test_data = train_test_data_setup(data, shuffle_flag=True)

    pipeline.fit(input_data=train_data)
    results = pipeline.predict(input_data=test_data)

    roc_auc_on_test = roc_auc(y_true=test_data.target,
                              y_score=results.predict,
                              multi_class='ovo',
                              average='macro')

    assert roc_auc_on_test > 0.95


if __name__ == '__main__':
    multiclassification_pipeline_fit_correct()
