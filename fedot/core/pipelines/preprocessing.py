from copy import deepcopy
from typing import Union, List, Optional

import numpy as np
import pandas as pd

from fedot.core.data.data import InputData, data_has_categorical_features, data_type_is_table
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_transformations import \
    OneHotEncodingImplementation, ImputationImplementation
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    DataOperationImplementation
from fedot.core.pipelines.node import Node

# The allowed empirical partition limit of the number of rows to delete.
# Rows that have 'string' type, instead of other 'integer' observes.
# Example: 90% objects in column are 'integer', other are 'string'. Then
# we will try to convert 'string' data to 'integer', otherwise delete it.
EMPIRICAL_PARTITION = 0.5


def imputation_implementation(data: Union[InputData, MultiModalData]) -> Union[InputData, MultiModalData]:
    if isinstance(data, InputData):
        return _imputation_implementation_unidata(data)
    if isinstance(data, MultiModalData):
        for data_source_name, values in data.items():
            if data_source_name.startswith('data_source_table') or data_source_name.startswith('data_source_ts'):
                data[data_source_name].features = _imputation_implementation_unidata(values)
        return data
    raise ValueError(f"Data format is not supported.")


def custom_preprocessing(data: Union[InputData, MultiModalData]):
    if isinstance(data, InputData):
        if data_type_is_table(data):
            data = _preprocessing_input_data(data)
    elif isinstance(data, MultiModalData):
        for data_source_name, values in data.items():
            if data_type_is_table(values):
                data[data_source_name] = _preprocessing_input_data(values)

    return data


def drop_features_full_of_nans(data: InputData):
    """ Dropping features with more than 30% nan's

    :param data: data to transform
    :return: transformed data
    """
    output = deepcopy(data)

    features = data.features
    n_samples = features.shape[0]
    relevant_feature = []
    t_features = np.transpose(features)

    for i, feature in enumerate(t_features):
        if np.sum(pd.isna(feature)) / n_samples < 0.3:
            relevant_feature.append(i)

    if relevant_feature is not None:
        output.features = np.transpose(t_features[relevant_feature])

    return output


def encode_data_for_fit(data: Union[InputData, MultiModalData]) -> \
        Union[List[DataOperationImplementation], DataOperationImplementation]:
    """ Encode categorical features to numerical. In additional,
    save encoders to use later for prediction data.

    :param data: data to transform
    :return encoders: operation preprocessing categorical features or list of it
    """

    encoders = None
    if isinstance(data, InputData):
        transformed, encoder = _create_encoder(data)
        encoders = encoder
        data.features = transformed
    elif isinstance(data, MultiModalData):
        encoders = {}
        for data_source_name, values in data.items():
            if data_source_name.startswith('data_source_table'):
                transformed, encoder = _create_encoder(values)
                if encoder is not None:
                    encoders[data_source_name] = encoder
                data[data_source_name].features = transformed

    return encoders


def encode_data_for_prediction(data: Union[InputData, MultiModalData],
                               encoders: Union[dict, DataOperationImplementation]):
    """ Transformation the prediction data inplace. Use the same transformations as for the training data.

    :param data: data to transformation
    :param encoders: encoders f transformation
    """
    if encoders:
        if isinstance(data, InputData):
            transformed = encoders.transform(data, True).predict
            data.features = transformed
        elif isinstance(data, MultiModalData):
            for data_source_name, encoder in encoders.items():
                transformed = encoder.transform(data[data_source_name], True).predict
                data[data_source_name].features = transformed


def pipeline_encoders_validation(pipeline) -> (bool, bool):
    """ Check whether Imputation and OneHotEncoder operation exist in pipeline.

    :param pipeline: pipeline to check
    :return (bool, bool): has Imputation and OneHotEncoder in pipeline
    """

    has_imputers, has_encoders = [], []

    def _check_imputer_encoder_recursion(root: Optional[Node], has_imputer: bool = False, has_encoder: bool = False):
        node_type = root.operation.operation_type
        if node_type == 'simple_imputation':
            has_imputer = True
        if node_type == 'one_hot_encoding':
            has_encoder = True

        if has_imputer and has_encoder:
            return has_imputer, has_encoder
        elif root.nodes_from is None:
            return has_imputer, has_encoder

        for node in root.nodes_from:
            answer = _check_imputer_encoder_recursion(node, has_imputer, has_encoder)
            if answer is not None:
                imputer, encoder = answer
                has_imputers.append(imputer)
                has_encoders.append(encoder)

    _check_imputer_encoder_recursion(pipeline.root_node)

    if not has_imputers and not has_encoders:
        return False, False

    has_imputer = len([_ for _ in has_imputers if not _]) == 0
    has_encoder = len([_ for _ in has_encoders if not _]) == 0
    return has_imputer, has_encoder


def is_np_array_has_nan(array):
    for x in array:
        if x is np.nan:
            return True
    return False


def remove_leading_trailing_spaces(data):
    """ Transform cells in columns from ' x ' to 'x' """
    features_df = pd.DataFrame(data.features)
    for column in features_df.columns:
        try:
            features_df[column] = features_df[column].str.strip()
        except AttributeError:
            # Column not a string and cannot be converted into str
            pass
    data.features = np.array(features_df)
    return data


def clean_data(data: Union[InputData, MultiModalData]) -> Union[InputData, MultiModalData]:
    """ Remove extra spaces from dataframe """
    if isinstance(data, InputData):
        data = remove_leading_trailing_spaces(data)
    elif isinstance(data, MultiModalData):
        for data_source_name, local_data in data.items():
            local_data = remove_leading_trailing_spaces(local_data)
            data.update({data_source_name: local_data})
    return data


def _create_encoder(data: InputData):
    """ Fills in the gaps, converts categorical features using OneHotEncoder and create encoder.

    :param data: data to preprocess
    :return tuple(array, Union[OneHotEncodingImplementation, None]): tuple of transformed and [encoder or None]
    """

    encoder = None
    if data_has_categorical_features(data):
        encoder = OneHotEncodingImplementation()
        encoder.fit(data)
        transformed = encoder.transform(data, True).predict
    else:
        transformed = data.features

    return transformed, encoder


def _imputation_implementation_unidata(data: InputData):
    """ Fill in the gaps in the data inplace.

    :param data: data for fill in the gaps
    """
    imputer = ImputationImplementation()
    output_data = imputer.fit_transform(data)
    transformed = InputData(features=output_data.predict, data_type=output_data.data_type,
                            target=output_data.target, task=output_data.task, idx=output_data.idx)
    return transformed


def _preprocessing_input_data(data: InputData) -> InputData:
    features = data.features
    target = data.target

    # delete rows with equal target None
    if target is not None and len(target.shape) != 0 and is_np_array_has_nan(target):
        target_index_with_nan = np.hstack(np.argwhere(np.isnan(target)))
        data.features = np.delete(features, target_index_with_nan, 0)
        data.target = np.delete(data.target, target_index_with_nan, 0)
        data.idx = np.delete(data.idx, target_index_with_nan, 0)

    return data


def _is_numeric(s) -> bool:
    """ Check if variable converted to float.

    :param s: any type variable
    :return: is variable convertable to float
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def _try_convert_to_numeric(values):
    try:
        values = pd.to_numeric(values)
        values = values.astype(np.number)
    except ValueError:
        pass
    return values
