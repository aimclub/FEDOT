from fedot.core.data.data_split import train_test_data_setup
from fedot.preprocessing.data_types import TableTypesCorrector
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline

from test.unit.preprocessing.test_pipeline_preprocessing import data_with_mixed_types_in_each_column


def generate_linear_pipeline():
    """
    Generate linear pipeline where the data changes from node to node.
    The number of columns increases (poly_features), the number of columns
    decreases (rfe_lin_class), the types change while keeping the number of
    columns ('label_encoding')
    """
    encoding_label = PrimaryNode('label_encoding')
    poly_node = SecondaryNode('poly_features', nodes_from=[encoding_label])
    rfe_node = SecondaryNode('rfe_lin_class', nodes_from=[poly_node])
    final_node = SecondaryNode('dt', nodes_from=[rfe_node])
    pipeline = Pipeline(final_node)

    return pipeline


def test_column_types_converting_correctly():
    """ Generate dataset with mixed data types per columns and then apply correction procedure.
    After converting into new types, field "column_types" for target and features is checked
    """
    data = data_with_mixed_types_in_each_column(multi_output=True)
    # Remove column full of nans (column with id 0)
    data.features = data.features[:, 1:]

    types_corr = TableTypesCorrector()
    data = types_corr.convert_data_for_fit(data)

    features_types = data.supplementary_data.column_types['features']
    target_types = data.supplementary_data.column_types['target']

    assert len(features_types) == len(target_types) == 2
    assert features_types[0] == "<class 'str'>"
    assert features_types[1] == "<class 'float'>"
    assert target_types[0] == target_types[0] == "<class 'str'>"


def test_column_types_process_correctly():
    """ Generate table with different types of data per columns.
    After transferring via pipeline, they must be processed correctly
    (added and removed)
    """

    data = data_with_mixed_types_in_each_column()
    train_data, test_data = train_test_data_setup(data, split_ratio=0.9)

    # Remove target from test sample
    test_data.target = None
    pipeline = generate_linear_pipeline()

    pipeline.fit(train_data)
    predicted = pipeline.predict(test_data)

    features_columns = predicted.supplementary_data.column_types['features']
    assert len(features_columns) == predicted.predict.shape[1]
    # All output values are float
    assert all('float' in str(feature_type) for feature_type in features_columns)
