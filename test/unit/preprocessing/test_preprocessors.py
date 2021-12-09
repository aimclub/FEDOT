from fedot.preprocessing.data_types import TableTypesCorrector
from test.unit.preprocessing.test_pipeline_preprocessing import data_with_mixed_types_in_each_column


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
