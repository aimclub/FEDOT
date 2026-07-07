from __future__ import annotations

import sys
from abc import abstractmethod
from functools import wraps
from typing import Optional, Tuple

import torch

from fedot.core.caching.predictions_cache import PredictionsCache
from fedot.core.data.tensor_data import TensorData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast
from fedot.core.utils import default_fedot_data_dir
from fedot.utilities.custom_errors import AbstractMethodNotImplementError
from fedot.utilities.debug import is_analytic_mode


def from_maximised_metric(metric_func):
    @wraps(metric_func)
    def wrapper(*args, **kwargs):
        return -metric_func(*args, **kwargs)

    return wrapper


class Metric:
    @classmethod
    @abstractmethod
    def get_value(cls, **kwargs) -> float:
        """ Get metrics value based on pipeline and other optional arguments. """
        raise AbstractMethodNotImplementError

# TODO @romankuklo: add validation schema

class QualityMetric(Metric):
    max_penalty_part = 0.01
    output_mode = 'default'
    default_value = 0

    @staticmethod
    @abstractmethod
    def metric(reference: TensorData, predicted: TensorData) -> float:
        raise AbstractMethodNotImplementError

    @staticmethod
    def _to_1d_tensor(value: torch.Tensor) -> torch.Tensor:
        value = value.detach()
        if value.ndim > 1 and value.shape[-1] == 1:
            value = value.reshape(-1)
        return value

    @staticmethod
    def _least_frequent_val(value: torch.Tensor) -> torch.Tensor:
        labels, counts = torch.unique(QualityMetric._to_1d_tensor(value), return_counts=True)
        return labels[torch.argmin(counts)]

    @classmethod
    def get_value(cls,
                  pipeline: Pipeline,
                  reference_data: TensorData,
                  validation_blocks: Optional[int] = None,
                  predictions_cache: Optional[PredictionsCache] = None,
                  fold_id: Optional[int] = None) -> float:
        """ Get metric value based on pipeline, reference data, and number of validation blocks.
        Args:
            pipeline: a :class:`Pipeline` instance for evaluation.
            reference_data: :class:`TensorData` for evaluation.
            validation_blocks: number of validation blocks. Used only for time series forecasting.
                If ``None``, data separation is not performed.
        """
        metric = cls.default_value
        try:
            if validation_blocks is not None:
                # TODO @artemlunev: add after validation_blocks implementation
                # reference_data, results = cls._in_sample_prediction(
                #     pipeline,
                #     reference_data,
                #     validation_blocks,
                #     predictions_cache=predictions_cache,
                #     fold_id=fold_id
                # )
                raise NotImplementedError('TensorData in-sample metric evaluation is not supported yet')
            reference_data, results = cls._simple_prediction(
                pipeline, reference_data, predictions_cache, fold_id)
            metric = cls.metric(reference_data, results)

            # TODO @romankuklo: add analytic mode
            # if is_analytic_mode():
            #     from fedot.core.data.visualisation import plot_forecast

            #     pipeline_id = str(uuid4())
            #     save_path = Path(default_fedot_data_dir(),
            #                      'ts_forecasting_debug', pipeline_id)
            #     if not os.path.exists(save_path):
            #         os.makedirs(save_path)
            #     pipeline.show(save_path=Path(save_path, 'pipeline.png'))
            #     plot_forecast(reference_data, results, in_sample=True,
            #                   title=f'Forecast with metric {round(metric, 4)}',
            #                   save_path=Path(save_path, 'forecast.png'))

        except Exception:
            pipeline.log.log_or_raise(
                'info', ValueError('Metric can not be evaluated'))

        return metric

    @classmethod
    def _simple_prediction(cls,
                           pipeline: Pipeline,
                           reference_data: TensorData,
                           predictions_cache: Optional[PredictionsCache] = None,
                           fold_id: Optional[int] = None) -> Tuple[TensorData, TensorData]:
        """Method calls pipeline.predict_tensordata() and returns the result."""
        return reference_data, pipeline.predict_tensordata(
            reference_data, output_mode=cls.output_mode, predictions_cache=predictions_cache, fold_id=fold_id)

    @classmethod
    def get_value_with_penalty(cls,
                               pipeline: Pipeline,
                               reference_data: TensorData,
                               validation_blocks: Optional[int] = None,
                               predictions_cache: Optional[PredictionsCache] = None,
                               fold_id: Optional[int] = None) -> float:
        quality_metric = cls.get_value(pipeline=pipeline,
                                       reference_data=reference_data,
                                       validation_blocks=validation_blocks,
                                       predictions_cache=predictions_cache,
                                       fold_id=fold_id)
        structural_metric = StructuralComplexity.get_value(pipeline)

        penalty = abs(structural_metric *
                      quality_metric * cls.max_penalty_part)
        metric_with_penalty = (quality_metric +
                               min(penalty, abs(quality_metric * cls.max_penalty_part)))
        return metric_with_penalty
    
    # TODO @romankuklo: add in-sample prediction for tensor data
    # @staticmethod
    # def _in_sample_prediction(pipeline: Pipeline,
    #                           data: InputData,
    #                           validation_blocks: int,
    #                           predictions_cache: Optional[PredictionsCache] = None,
    #                           fold_id: Optional[int] = None) -> Tuple[InputData, OutputData]:
    #     """ Performs in-sample pipeline validation for time series prediction """

    #     horizon = int(validation_blocks *
    #                   data.task.task_params.forecast_length)

    #     actual_values = data.target[-horizon:]

    #     predicted_values = in_sample_ts_forecast(pipeline=pipeline,
    #                                              input_data=data,
    #                                              horizon=horizon,
    #                                              predictions_cache=predictions_cache,
    #                                              fold_id=fold_id)

    #     # Wrap target and prediction arrays into OutputData and InputData
    #     results = OutputData(idx=np.arange(0, len(predicted_values)), features=predicted_values,
    #                          predict=predicted_values, task=data.task, target=predicted_values,
    #                          data_type=DataTypesEnum.ts)
    #     reference_data = InputData(idx=np.arange(0, len(actual_values)), features=actual_values,
    #                                task=data.task, target=actual_values, data_type=DataTypesEnum.ts)

    #     return reference_data, results


class RMSE(QualityMetric):
    default_value = sys.maxsize

    @staticmethod
    def metric(reference: TensorData, predicted: TensorData) -> float:
        y_true = reference.target.detach().to(dtype=torch.float64)
        y_pred = predicted.predict.detach().to(dtype=torch.float64)
        return float(torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item())


class MSE(QualityMetric):
    default_value = sys.maxsize

    @staticmethod
    def metric(reference: TensorData, predicted: TensorData) -> float:
        y_true = reference.target.detach().to(dtype=torch.float64)
        y_pred = predicted.predict.detach().to(dtype=torch.float64)
        return float(torch.mean((y_true - y_pred) ** 2).item())


class MSLE(QualityMetric):
    default_value = sys.maxsize

    @staticmethod
    def metric(reference: TensorData, predicted: TensorData) -> float:
        y_true = reference.target.detach().to(dtype=torch.float64)
        y_pred = predicted.predict.detach().to(dtype=torch.float64)
        return float(torch.mean((torch.log1p(y_true) - torch.log1p(y_pred)) ** 2).item())


class MAPE(QualityMetric):
    default_value = sys.maxsize

    @staticmethod
    def metric(reference: TensorData, predicted: TensorData) -> float:
        y_true = reference.target.detach().to(dtype=torch.float64)
        y_pred = predicted.predict.detach().to(dtype=torch.float64)
        eps = torch.finfo(y_true.dtype).eps
        denominator = torch.maximum(torch.abs(y_true), torch.tensor(eps, device=y_true.device))
        return float(torch.mean(torch.abs(y_true - y_pred) / denominator).item())


class SMAPE(QualityMetric):
    default_value = sys.maxsize

    @staticmethod
    def metric(reference: TensorData, predicted: TensorData) -> float:
        y_true = reference.target.detach().to(dtype=torch.float64)
        y_pred = predicted.predict.detach().to(dtype=torch.float64)
        numerator = 2 * torch.abs(y_true - y_pred)
        denominator = torch.abs(y_true) + torch.abs(y_pred)
        result = numerator / denominator
        result = torch.where(torch.isnan(result), torch.zeros_like(result), result)
        return float(torch.mean(100 * result).item())


class F1(QualityMetric):
    default_value = 0
    output_mode = 'labels'
    binary_averaging_mode = 'binary'
    multiclass_averaging_mode = 'weighted'

    @staticmethod
    def _binary_precision(y_true: torch.Tensor, y_pred: torch.Tensor, pos_label: torch.Tensor) -> torch.Tensor:
        positive_pred = y_pred == pos_label
        true_positive = torch.sum((y_true == pos_label) & positive_pred).to(torch.float64)
        predicted_positive = torch.sum(positive_pred).to(torch.float64)
        if predicted_positive == 0:
            return torch.tensor(0.0, dtype=torch.float64, device=y_true.device)
        return true_positive / predicted_positive

    @staticmethod
    def _binary_recall(y_true: torch.Tensor, y_pred: torch.Tensor, pos_label: torch.Tensor) -> torch.Tensor:
        positive_true = y_true == pos_label
        true_positive = torch.sum(positive_true & (y_pred == pos_label)).to(torch.float64)
        actual_positive = torch.sum(positive_true).to(torch.float64)
        if actual_positive == 0:
            return torch.tensor(0.0, dtype=torch.float64, device=y_true.device)
        return true_positive / actual_positive

    @staticmethod
    def _binary_f1(y_true: torch.Tensor, y_pred: torch.Tensor, pos_label: torch.Tensor) -> torch.Tensor:
        precision = F1._binary_precision(y_true, y_pred, pos_label)
        recall = F1._binary_recall(y_true, y_pred, pos_label)
        denominator = precision + recall
        if denominator == 0:
            return torch.tensor(0.0, dtype=torch.float64, device=y_true.device)
        return 2 * precision * recall / denominator

    @staticmethod
    def _multiclass_f1(y_true: torch.Tensor, y_pred: torch.Tensor, average: str) -> float:
        labels = torch.unique(y_true)
        values = []
        weights = []
        for label in labels:
            values.append(F1._binary_f1(y_true, y_pred, label))
            weights.append(torch.sum(y_true == label).to(torch.float64))
        values = torch.stack(values)
        weights = torch.stack(weights)
        if average == 'weighted':
            return float((values * weights / torch.sum(weights)).sum().item())
        return float(torch.mean(values).item())

    @staticmethod
    @from_maximised_metric
    def metric(reference: TensorData, predicted: TensorData) -> float:
        y_true = QualityMetric._to_1d_tensor(reference.target)
        y_pred = QualityMetric._to_1d_tensor(predicted.predict)
        if len(torch.unique(y_true)) == 2:
            return float(F1._binary_f1(
                y_true, y_pred, QualityMetric._least_frequent_val(reference.target)).item())
        return F1._multiclass_f1(y_true, y_pred, F1.multiclass_averaging_mode)


class MAE(QualityMetric):
    default_value = sys.maxsize

    @staticmethod
    def metric(reference: TensorData, predicted: TensorData) -> float:
        y_true = reference.target.detach().to(dtype=torch.float64)
        y_pred = predicted.predict.detach().to(dtype=torch.float64)
        return float(torch.mean(torch.abs(y_true - y_pred)).item())


class MASE(QualityMetric):
    default_value = sys.maxsize

    @staticmethod
    def metric(reference: TensorData, predicted: TensorData) -> float:
        y_true = reference.target.detach().to(dtype=torch.float64)
        y_pred = predicted.predict.detach().to(dtype=torch.float64)
        y_train = reference.features.detach().to(dtype=torch.float64)
        denominator = torch.mean(torch.abs(y_train[1:] - y_train[:-1]))
        if denominator == 0:
            raise ValueError('MASE is undefined for constant history')
        return float((torch.mean(torch.abs(y_true - y_pred)) / denominator).item())


class R2(QualityMetric):
    default_value = 0

    @staticmethod
    def metric(reference: TensorData, predicted: TensorData) -> float:
        y_true = reference.target.detach().to(dtype=torch.float64)
        y_pred = predicted.predict.detach().to(dtype=torch.float64)
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)
        residual = torch.sum((y_true - y_pred) ** 2, dim=0)
        total = torch.sum((y_true - torch.mean(y_true, dim=0)) ** 2, dim=0)
        scores = torch.where(total == 0, torch.zeros_like(total), 1 - residual / total)
        return float(torch.mean(scores).item())


class ROCAUC(QualityMetric):
    default_value = 0.5

    @staticmethod
    def _rankdata_average(value: torch.Tensor) -> torch.Tensor:
        sorted_value, order = torch.sort(value)
        ranks = torch.empty_like(value, dtype=torch.float64)
        start = 0
        while start < len(value):
            end = start + 1
            while end < len(value) and sorted_value[end] == sorted_value[start]:
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2.0
            start = end
        return ranks

    @staticmethod
    def _binary_roc_auc(y_true: torch.Tensor, y_score: torch.Tensor) -> torch.Tensor:
        y_true = y_true.reshape(-1).to(dtype=torch.bool)
        y_score = y_score.detach().to(dtype=torch.float64).reshape(-1)
        positive_count = torch.sum(y_true).to(torch.float64)
        negative_count = torch.sum(~y_true).to(torch.float64)
        if positive_count == 0 or negative_count == 0:
            raise ValueError('Only one class present in y_true. ROC AUC score is not defined.')
        ranks = ROCAUC._rankdata_average(y_score)
        positive_rank_sum = torch.sum(ranks[y_true])
        return (positive_rank_sum - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count)

    @staticmethod
    def _roc_auc_score(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
        y_true = QualityMetric._to_1d_tensor(y_true)
        labels = torch.unique(y_true)
        if len(labels) == 2:
            positive_label = labels[-1]
            if y_score.ndim > 1:
                y_score = y_score[:, -1]
            return float(ROCAUC._binary_roc_auc(y_true == positive_label, y_score).item())
        auc_values = [
            ROCAUC._binary_roc_auc(y_true == label, y_score[:, class_idx])
            for class_idx, label in enumerate(labels)
        ]
        return float(torch.mean(torch.stack(auc_values)).item())

    @staticmethod
    @from_maximised_metric
    def metric(reference: TensorData, predicted: TensorData) -> float:
        return ROCAUC._roc_auc_score(reference.target, predicted.predict)


class Precision(QualityMetric):
    output_mode = 'labels'
    binary_averaging_mode = 'binary'
    multiclass_averaging_mode = 'macro'

    @staticmethod
    def _binary_precision(y_true: torch.Tensor, y_pred: torch.Tensor, pos_label: torch.Tensor) -> torch.Tensor:
        positive_pred = y_pred == pos_label
        true_positive = torch.sum((y_true == pos_label) & positive_pred).to(torch.float64)
        predicted_positive = torch.sum(positive_pred).to(torch.float64)
        if predicted_positive == 0:
            return torch.tensor(0.0, dtype=torch.float64, device=y_true.device)
        return true_positive / predicted_positive

    @staticmethod
    def _multiclass_precision(y_true: torch.Tensor, y_pred: torch.Tensor, average: str) -> float:
        labels = torch.unique(y_true)
        values = []
        weights = []
        for label in labels:
            values.append(Precision._binary_precision(y_true, y_pred, label))
            weights.append(torch.sum(y_true == label).to(torch.float64))
        values = torch.stack(values)
        weights = torch.stack(weights)
        if average == 'weighted':
            return float((values * weights / torch.sum(weights)).sum().item())
        return float(torch.mean(values).item())

    @staticmethod
    @from_maximised_metric
    def metric(reference: TensorData, predicted: TensorData) -> float:
        y_true = QualityMetric._to_1d_tensor(reference.target)
        y_pred = QualityMetric._to_1d_tensor(predicted.predict)
        if len(torch.unique(y_true)) > 2:
            return Precision._multiclass_precision(y_true, y_pred, Precision.multiclass_averaging_mode)
        return float(Precision._binary_precision(
            y_true, y_pred, QualityMetric._least_frequent_val(reference.target)).item())


class Logloss(QualityMetric):
    default_value = sys.maxsize

    @staticmethod
    def metric(reference: TensorData, predicted: TensorData) -> float:
        y_true = reference.target.detach()
        if y_true.ndim > 1 and y_true.shape[-1] == 1:
            y_true = y_true.reshape(-1)
        y_true = y_true.to(dtype=torch.long)
        y_pred = predicted.predict.detach().to(dtype=torch.float64)
        eps = torch.finfo(y_pred.dtype).eps
        y_pred = torch.clamp(y_pred, eps, 1 - eps)
        if y_pred.ndim == 1:
            loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
            return float(torch.mean(loss).item())
        y_pred = y_pred / torch.sum(y_pred, dim=1, keepdim=True)
        loss = -torch.log(y_pred[torch.arange(len(y_true), device=y_pred.device), y_true])
        return float(torch.mean(loss).item())


class Accuracy(QualityMetric):
    default_value = 0
    output_mode = 'labels'

    @staticmethod
    @from_maximised_metric
    def metric(reference: TensorData, predicted: TensorData) -> float:
        y_true = reference.target.detach()
        y_pred = predicted.predict.detach()
        if y_true.ndim > 1 and y_true.shape[-1] == 1:
            y_true = y_true.reshape(-1)
        if y_pred.ndim > 1 and y_pred.shape[-1] == 1:
            y_pred = y_pred.reshape(-1)
        return float(torch.mean(
            (y_true == y_pred).to(torch.float64)).item())


class Silhouette(QualityMetric):
    default_value = 1

    @staticmethod
    @from_maximised_metric
    def metric(reference: TensorData, predicted: TensorData) -> float:
        features = reference.features.detach().to(dtype=torch.float64)
        labels = predicted.predict.detach()
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        if labels.ndim > 1 and labels.shape[-1] == 1:
            labels = labels.reshape(-1)

        unique_labels = torch.unique(labels)
        if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
            raise ValueError('Silhouette score is defined for at least 2 and less than n_samples labels')

        distances = torch.cdist(features, features)
        sample_scores = []
        for sample_idx, label in enumerate(labels):
            same_cluster = labels == label
            same_cluster[sample_idx] = False
            if torch.any(same_cluster):
                intra_cluster_distance = torch.mean(distances[sample_idx, same_cluster])
            else:
                sample_scores.append(torch.tensor(0.0, dtype=torch.float64, device=features.device))
                continue

            other_cluster_distances = []
            for other_label in unique_labels:
                if other_label == label:
                    continue
                other_cluster_distances.append(torch.mean(distances[sample_idx, labels == other_label]))
            nearest_cluster_distance = torch.min(torch.stack(other_cluster_distances))
            denominator = torch.maximum(intra_cluster_distance, nearest_cluster_distance)
            sample_scores.append((nearest_cluster_distance - intra_cluster_distance) / denominator)

        return float(torch.mean(torch.stack(sample_scores)).item())


class ComplexityMetric(Metric):
    default_value = 0
    norm_constant = 1

    @classmethod
    def get_value(cls, pipeline: Pipeline, **kwargs) -> float:
        """ Get metric value and apply norm_constant to it. """
        return cls.metric(pipeline, **kwargs) / cls.norm_constant

    @classmethod
    @abstractmethod
    def metric(cls, pipeline: Pipeline, **kwargs) -> float:
        """ Get metrics value based on pipeline. """
        raise AbstractMethodNotImplementError


class StructuralComplexity(ComplexityMetric):
    norm_constant = 30

    @classmethod
    def metric(cls, pipeline: Pipeline, **kwargs) -> float:
        return pipeline.depth ** 2 + pipeline.length


class NodeNum(ComplexityMetric):
    norm_constant = 10

    @classmethod
    def metric(cls, pipeline: Pipeline, **kwargs) -> float:
        return pipeline.length


class ComputationTime(ComplexityMetric):
    default_value = sys.maxsize

    @classmethod
    def metric(cls, pipeline: Pipeline, **kwargs) -> float:
        return pipeline.computation_time
