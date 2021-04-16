import sys
from abc import abstractmethod
from copy import copy

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, log_loss, mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, mean_squared_log_error, precision_score, r2_score, roc_auc_score,
                             silhouette_score)

from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.tasks import TaskTypesEnum


def from_maximised_metric(metric_func):
    def wrapper(*args, **kwargs):
        return -metric_func(*args, **kwargs)

    return wrapper


class Metric:
    output_mode = 'default'
    default_value = 0

    @classmethod
    @abstractmethod
    def get_value(cls, chain: Chain, reference_data: InputData) -> float:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def metric(reference: InputData, predicted: OutputData) -> float:
        raise NotImplementedError()


class QualityMetric:
    max_penalty_part = 0.01
    output_mode = 'default'
    default_value = 0

    @classmethod
    def get_value(cls, chain: Chain, reference_data: InputData) -> float:
        metric = cls.default_value
        try:
            results = chain.predict(reference_data, output_mode=cls.output_mode)

            if reference_data.task.task_type == TaskTypesEnum.ts_forecasting:
                # Convert prediction into one-dimensional array
                forecast_values = np.ravel(np.array(results.predict))
                results.predict = forecast_values
                metric = cls.metric(reference_data, results)
            else:
                metric = cls.metric(reference_data, results)
        except Exception as ex:
            print(f'Metric evaluation error: {ex}')
        return metric

    @classmethod
    def get_value_with_penalty(cls, chain: Chain, reference_data: InputData) -> float:
        quality_metric = cls.get_value(chain, reference_data)
        structural_metric = StructuralComplexity.get_value(chain)

        penalty = abs(structural_metric * quality_metric * cls.max_penalty_part)
        metric_with_penalty = (quality_metric +
                               min(penalty, abs(quality_metric * cls.max_penalty_part)))
        return metric_with_penalty

    @staticmethod
    @abstractmethod
    def metric(reference: InputData, predicted: OutputData) -> float:
        raise NotImplementedError()


class RMSE(QualityMetric):
    default_value = sys.maxsize

    @staticmethod
    def metric(reference: InputData, predicted: OutputData) -> float:
        return mean_squared_error(y_true=reference.target,
                                  y_pred=predicted.predict, squared=False)


class MSE(QualityMetric):
    default_value = sys.maxsize

    @staticmethod
    def metric(reference: InputData, predicted: OutputData) -> float:
        return mean_squared_error(y_true=reference.target,
                                  y_pred=predicted.predict, squared=True)


class MSLE(QualityMetric):
    default_value = sys.maxsize

    @staticmethod
    def metric(reference: InputData, predicted: OutputData) -> float:
        return mean_squared_log_error(y_true=reference.target,
                                      y_pred=predicted.predict)


class MAPE(QualityMetric):
    default_value = sys.maxsize

    @staticmethod
    def metric(reference: InputData, predicted: OutputData) -> float:
        return mean_absolute_percentage_error(y_true=reference.target,
                                              y_pred=predicted.predict)


class F1(QualityMetric):
    default_value = 0
    output_mode = 'labels'

    @staticmethod
    @from_maximised_metric
    def metric(reference: InputData, predicted: OutputData) -> float:
        n_classes = reference.num_classes
        if n_classes > 2:
            additional_params = {'average': 'macro'}
        else:
            additional_params = {}
        return f1_score(y_true=reference.target, y_pred=predicted.predict,
                        **additional_params)


class MAE(QualityMetric):
    default_value = sys.maxsize

    @staticmethod
    def metric(reference: InputData, predicted: OutputData) -> float:
        return mean_absolute_error(y_true=reference.target, y_pred=predicted.predict)


class R2(QualityMetric):
    default_value = 0

    @staticmethod
    def metric(reference: InputData, predicted: OutputData) -> float:
        return r2_score(y_true=reference.target, y_pred=predicted.predict)


class ROCAUC(QualityMetric):
    default_value = 0.5

    @staticmethod
    @from_maximised_metric
    def metric(reference: InputData, predicted: OutputData) -> float:

        n_classes = reference.num_classes
        if n_classes > 2:
            additional_params = {'multi_class': 'ovo', 'average': 'macro'}
        else:
            additional_params = {}

        score = round(roc_auc_score(y_score=predicted.predict,
                                    y_true=reference.target,
                                    **additional_params), 3)
        return score


class Precision(QualityMetric):
    output_mode = 'labels'

    @staticmethod
    @from_maximised_metric
    def metric(reference: InputData, predicted: OutputData) -> float:
        return precision_score(y_true=reference.target, y_pred=predicted.predict)


class Logloss(QualityMetric):
    default_value = sys.maxsize

    @staticmethod
    def metric(reference: InputData, predicted: OutputData) -> float:
        return log_loss(y_true=reference.target, y_pred=predicted.predict)


class Accuracy(QualityMetric):
    default_value = 0
    output_mode = 'labels'

    @staticmethod
    @from_maximised_metric
    def metric(reference: InputData, predicted: OutputData) -> float:
        return accuracy_score(y_true=reference.target, y_pred=predicted.predict)


class Silhouette(QualityMetric):
    default_value = 1

    @staticmethod
    @from_maximised_metric
    def metric(reference: InputData, predicted: OutputData) -> float:
        return silhouette_score(reference.features, labels=predicted.predict)


class StructuralComplexity(Metric):
    @classmethod
    def get_value(cls, chain: Chain, **args) -> float:
        norm_constant = 30
        return (chain.depth ** 2 + chain.length) / norm_constant


class NodeNum(Metric):
    @classmethod
    def get_value(cls, chain: Chain, **args) -> float:
        norm_constant = 10
        return chain.length / norm_constant


class ComputationTime(Metric):
    @classmethod
    def get_value(cls, chain: Chain, **args) -> float:
        return chain.computation_time
