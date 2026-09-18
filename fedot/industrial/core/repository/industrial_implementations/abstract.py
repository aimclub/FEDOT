from copy import deepcopy
from functools import partial
from itertools import chain
from typing import List, Optional, Union

from fedot.core.constants import default_data_split_ratio_by_task
from fedot.core.data.common.array_utils import atleast_4d
from fedot.core.data.split.cv_folds import cv_generator
from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.data.split.data_split import _split_input_data_by_indexes
from fedot.core.data.merge.data_merger import TSDataMerger, DataMerger, ImageDataMerger, TextDataMerger
from fedot.core.data.multimodal.multi_modal import MultiModalData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.optimisers.objective import DataSource
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.preprocessing.data_types import TYPE_TO_ID
from joblib import delayed, Parallel
from sklearn.model_selection import train_test_split

from fedot.industrial.core.architecture.preprocessing.data_convertor import NumpyConverter
from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.tuning.search_space import get_industrial_search_space


def split_time_series_industrial(data: InputData,
                                 validation_blocks: Optional[int] = None,
                                 **kwargs):
    """ Split time series data into train and test parts

    :param data: InputData object to split
    :param validation_blocks: validation blocks are used for test
    """
    if isinstance(data.task.task_params, dict):
        forecast_length = data.task.task_params['forecast_length']
    else:
        forecast_length = data.task.task_params.forecast_length
    if validation_blocks is not None:
        forecast_length *= validation_blocks
    train_data = deepcopy(data)
    train_data.target = data.features[-forecast_length:]
    train_data.features = data.features[:-forecast_length]
    return train_data, data


def split_any_industrial(data: InputData,
                         split_ratio: float,
                         shuffle: bool,
                         stratify: bool,
                         random_seed: int,
                         **kwargs):
    """ Split any data except timeseries into train and test parts

    :param data: InputData object to split
    :param split_ratio: share of train data between 0 and 1
    :param shuffle: is data needed to be shuffled or not
    :param stratify: make stratified sample or not
    :param random_seed: random_seed for shuffle
    """

    stratify_labels = data.target if stratify else None

    def __split_loop(data, ratio, shuffle, stratify_labels):
        train_ids, test_ids = train_test_split(np.arange(0, len(data.target)),
                                               test_size=1 - ratio,
                                               shuffle=shuffle,
                                               random_state=random_seed,
                                               stratify=stratify_labels)
        is_clf_task = data.task.task_type.name.__contains__('classification')
        train_data = _split_input_data_by_indexes(data, index=train_ids)
        test_data = _split_input_data_by_indexes(data, index=test_ids)
        if is_clf_task:
            correct_split = np.unique(test_data.target).shape[0] == np.unique(
                train_data.target).shape[0]
        else:
            correct_split = True
        return train_data, test_data, correct_split

    for ratio in [split_ratio, 0.6, 0.5, 0.4, 0.3, 0.1]:
        train_data, test_data, correct_split = __split_loop(
            data, ratio, shuffle, stratify_labels)
        if correct_split:
            break
    return train_data, test_data


def _are_stratification_allowed(
        data: Union[InputData, MultiModalData], split_ratio: float) -> bool:
    """ Check that stratification may be done
        :param data: data for split
        :param split_ratio: relation between train data length and all data length
        :return bool: stratification is allowed"""

    # check task_type
    if data.task.task_type is not TaskTypesEnum.classification:
        return False
    else:
        return True


def _are_cv_folds_allowed(
        data: Union[InputData, MultiModalData], split_ratio: float, cv_folds: int):
    try:
        # fast way
        classes = np.unique(data.target, return_counts=True)
    except Exception:
        # slow way
        from collections import Counter
        classes = Counter(data.target)
        classes = [list(classes), list(classes.values())]

    # check that there are enough labels for two samples
    if not all(x > 1 for x in classes[1]):
        if __debug__:
            # tests often use very small datasets that are not suitable for data splitting
            # stratification is disabled for tests
            return None
        else:
            raise ValueError(
                ("There is the only value for some classes:"
                 f" {', '.join(str(val) for val, count in zip(*classes) if count == 1)}."
                 f" Data split can not be done for {data.task.task_type.name} task."))

    # check that split ratio allows to set all classes to both samples
    test_size = round(len(data.target) * (1. - split_ratio))
    labels_count = len(classes[0])
    if test_size < labels_count:
        return None
    else:
        return cv_folds


def build_industrial(self, data: Union[InputData, MultiModalData]) -> DataSource:
    self.split_ratio = self.split_ratio or default_data_split_ratio_by_task[
        data.task.task_type]

    # Check cv_folds
    if self.cv_folds is not None:
        try:
            self.cv_folds = int(self.cv_folds)
        except ValueError:
            raise ValueError(f"cv_folds is not integer: {self.cv_folds}")
        if self.cv_folds < 2:
            self.cv_folds = None
        if self.cv_folds > data.target.shape[0] - 1:
            raise ValueError(
                (f"cv_folds ({self.cv_folds}) is greater than"
                 f" the maximum allowed count {data.target.shape[0] - 1}"))

    # Calculate the number of validation blocks for timeseries forecasting
    if data.task.task_type is TaskTypesEnum.ts_forecasting and self.validation_blocks is None:
        current_split_ratio = self.split_ratio
        # workaround for compability with basic Fedot
        # copy_input = deepcopy(data)
        data.target = data.features
        self._propose_cv_folds_and_validation_blocks(data)
        if self.cv_folds is None:
            self.split_ratio = current_split_ratio

    # Check split_ratio
    if self.cv_folds is None and not (0 < self.split_ratio < 1):
        raise ValueError(
            f'split_ratio is {self.split_ratio} but should be between 0 and 1')

    if data.task.task_type is not TaskTypesEnum.ts_forecasting and self.stratify:
        # check that stratification can be done
        # for cross validation split ratio is defined as validation_size /
        # all_data_size
        split_ratio = self.split_ratio if self.cv_folds is None else (
            1 - 1 / (self.cv_folds + 1))
        self.stratify = _are_stratification_allowed(data, split_ratio)
        self.cv_folds = _are_cv_folds_allowed(data, split_ratio, self.cv_folds)
        if not self.stratify:
            self.log.info("Stratificated splitting of data is disabled.")

    # Stratification can not be done without shuffle
    self.shuffle |= self.stratify

    # Random seed depends on shuffle
    self.random_seed = (self.random_seed or 42) if self.shuffle else None

    # Split data
    if self.cv_folds is not None:
        self.log.info("K-folds cross validation is applied.")
        data_producer = partial(cv_generator,
                                data=data,
                                shuffle=self.shuffle,
                                cv_folds=self.cv_folds,
                                random_seed=self.random_seed,
                                stratify=self.stratify,
                                validation_blocks=self.validation_blocks)
    else:
        self.log.info("Hold out validation is applied.")
        data_producer = self._build_holdout_producer(data)

    return data_producer


def build_tuner(self, model_to_tune, tuning_params, train_data):
    def _create_tuner(tuning_params, tuning_data):
        custom_search_space = get_industrial_search_space(self)
        search_space = PipelineSearchSpace(custom_search_space=custom_search_space,
                                           replace_default_search_space=True)
        pipeline_tuner = TunerBuilder(train_data.task). \
            with_search_space(search_space). \
            with_tuner(tuning_params['tuner']). \
            with_cv_folds(tuning_params.get('cv_folds', None)). \
            with_n_jobs(tuning_params.get('n_jobs', 1)). \
            with_metric(tuning_params['metric']). \
            with_iterations(tuning_params.get('tuning_iterations', 50)). \
            build(tuning_data)

        return pipeline_tuner

    pipeline_tuner = _create_tuner(tuning_params, train_data)
    model_to_tune = pipeline_tuner.tune(model_to_tune)
    model_to_tune.fit(train_data)
    return model_to_tune


def postprocess_industrial_predicts(self, merged_predicts: np.array) -> np.array:
    """ Post-process merged predictions (e.g. reshape). """
    return merged_predicts


def update_column_types_industrial(self, output_data: OutputData):
    """Update column types after lagged transformation. All features becomes ``float``
    """

    _, features_n_cols, _ = output_data.predict.shape
    feature_type_ids = np.array([TYPE_TO_ID[float]] * features_n_cols)
    col_type_ids = {'features': feature_type_ids}

    if output_data.target is not None and len(output_data.target.shape) > 1:
        _, target_n_cols = output_data.target.shape
        target_type_ids = np.array([TYPE_TO_ID[float]] * target_n_cols)
        col_type_ids['target'] = target_type_ids
    output_data.supplementary_data.col_type_ids = col_type_ids


def preprocess_industrial_predicts(*args) -> List[np.array]:
    predicts = args[1]
    if len(predicts[0].shape) <= 3:
        return predicts
    else:
        reshaped_predicts = list(map(atleast_4d, predicts))

        # And check image sizes
        img_wh = [predict.shape[1:3] for predict in reshaped_predicts]
        # Can merge only images of the same size
        invalid_sizes = len(set(img_wh)) > 1
        if invalid_sizes:
            raise ValueError(
                "Can't merge images of different sizes: " + str(img_wh))
        return reshaped_predicts


def find_main_output_industrial(outputs: List['OutputData']) -> 'OutputData':
    """ Returns first output with main target or (if there are
    no main targets) the output with priority secondary target. """
    combine_ts_and_multi_ts = outputs[0].data_type.value.__contains__(
        'time') and len(outputs) != 1
    if combine_ts_and_multi_ts:
        try:
            priority_output = [
                x for x in outputs if len(x.target.shape) < 2][0]
        except Exception:
            # [x for x in outputs if len(x.target.shape) < 2][0]
            priority_output = outputs[0]
    else:
        priority_output = next((output for output in outputs
                                if output.supplementary_data.is_main_target), None)
        if not priority_output:
            flow_lengths = [
                output.supplementary_data.data_flow_length for output in outputs]
            i_priority_secondary = np.argmin(flow_lengths)
            priority_output = outputs[i_priority_secondary]
    return priority_output


def get_merger_industrial(outputs: List['OutputData']) -> 'DataMerger':
    """ Construct appropriate data merger for the outputs. """

    # Ensure outputs can be merged
    list_of_datatype = [output.data_type for output in outputs]
    if DataTypesEnum.ts in list_of_datatype:
        for output in outputs:
            output.data_type = DataTypesEnum.ts
    data_type = DataMerger.get_datatype_for_merge(
        output.data_type for output in outputs)
    if data_type is None:
        raise ValueError("Can't merge different data types")

    merger_by_type = {
        DataTypesEnum.table: DataMerger,
        DataTypesEnum.ts: TSDataMerger,
        DataTypesEnum.multi_ts: TSDataMerger,
        DataTypesEnum.image: ImageDataMerger,
        DataTypesEnum.text: TextDataMerger,
    }
    cls = merger_by_type.get(data_type)
    if not cls:
        raise ValueError(f'Unable to merge data type {cls}')
    return cls(outputs, data_type)


def merge_industrial_targets(self) -> np.array:
    filtered_main_target = self.main_output.target
    return filtered_main_target


def merge_industrial_predicts(*args) -> np.array:
    predicts = args[1]
    predicts = [NumpyConverter(
        data=prediction).convert_to_torch_format() for prediction in predicts]
    sample_shape, channel_shape, elem_shape = [
        (x.shape[0], x.shape[1], x.shape[2]) for x in predicts][0]

    sample_wise_concat = [x.shape[0] == sample_shape for x in predicts]
    chanel_concat = [x.shape[1] == channel_shape for x in predicts]
    element_wise_concat = [x.shape[2] == elem_shape for x in predicts]

    channel_match = all(chanel_concat)
    element_match = all(element_wise_concat)
    sample_match = all(sample_wise_concat)
    if sample_match and element_match:
        predict = np.concatenate(predicts, axis=1)
    elif sample_match and channel_match:
        predict = np.concatenate(predicts, axis=2)
    else:
        prediction_2d = np.concatenate(
            [x.reshape(x.shape[0], x.shape[1] * x.shape[2]) for x in predicts], axis=1)
        predict = prediction_2d.reshape(
            prediction_2d.shape[0], 1, prediction_2d.shape[1])

    return predict


def fit_topo_extractor_industrial(self, input_data: InputData):
    input_data.features = input_data.features if len(
        input_data.features.shape) == 0 else input_data.features.reshape(1, -1)
    self._window_size = int(
        input_data.features.shape[1] *
        self.window_size_as_share)
    self._window_size = max(self._window_size, 2)
    self._window_size = min(
        self._window_size,
        input_data.features.shape[1] - 2)
    return self


def transform_topo_extractor_industrial(self, input_data: InputData) -> OutputData:
    features = input_data.features if len(input_data.features.shape) == 0 \
        else input_data.features.reshape(1, -1)
    with Parallel(n_jobs=self.n_jobs, prefer='processes') as parallel:
        topological_features = parallel(delayed(self._extract_features)
                                        (np.mean(
                                            features[i:i + 2, ::self.stride], axis=0))
                                        for i in range(0, features.shape[0], 2))
    if len(topological_features) * 2 < features.shape[0]:
        topological_features.append(topological_features[-1])
    result = np.array(
        list(chain(*zip(topological_features, topological_features))))
    if result.shape[0] > features.shape[0]:
        result = result[:-1, :]
    np.nan_to_num(result, copy=False, nan=0, posinf=0, neginf=0)
    return result


def predict_operation_industrial(
        self,
        fitted_operation,
        data: InputData,
        params: Optional[OperationParameters] = None,
        output_mode: str = 'default',
        is_fit_stage: bool = False,
        *args,
        **kwargs):
    is_main_target = data.supplementary_data.is_main_target
    data_flow_length = data.supplementary_data.data_flow_length
    self._init(data.task, output_mode=output_mode, params=params,
               n_samples_data=data.features.shape[0])

    if is_fit_stage:
        prediction = self._eval_strategy.predict_for_fit(
            trained_operation=fitted_operation,
            predict_data=data,
            output_mode=output_mode)
    else:
        prediction = self._eval_strategy.predict(
            trained_operation=fitted_operation,
            predict_data=data,
            output_mode=output_mode)
    try:
        prediction.predict = prediction.predict.detach().numpy()
    except Exception:
        _ = 1
    prediction = self.assign_tabular_column_types(prediction, output_mode)

    # any inplace operations here are dangerous!
    if is_main_target is False:
        prediction.supplementary_data.is_main_target = is_main_target

    prediction.supplementary_data.data_flow_length = data_flow_length
    return prediction


def predict_industrial(self,
                       fitted_operation,
                       data: InputData,
                       params: Optional[Union[OperationParameters,
                                              dict]] = None,
                       output_mode: str = 'labels',
                       *args,
                       **kwargs):
    """This method is used for defining and running of the evaluation strategy
    to predict with the data provided

    Args:
        fitted_operation: trained operation object
        data: data used for prediction
        params: hyperparameters for operation
        output_mode: string with information about output of operation,
        for example, is the operation predict probabilities or class labels
    """
    return self._predict(
        fitted_operation,
        data,
        params,
        output_mode,
        is_fit_stage=False)


def predict_for_fit_industrial(
        self,
        fitted_operation,
        data: InputData,
        params: Optional[OperationParameters] = None,
        output_mode: str = 'default',
        *args,
        **kwargs):
    """This method is used for defining and running of the evaluation strategy
    to predict with the data provided during fit stage

    Args:
        fitted_operation: trained operation object
        data: data used for prediction
        params: hyperparameters for operation
        output_mode: string with information about output of operation,
            for example, is the operation predict probabilities or class labels
    """
    return self._predict(
        fitted_operation,
        data,
        params,
        output_mode,
        is_fit_stage=True)
