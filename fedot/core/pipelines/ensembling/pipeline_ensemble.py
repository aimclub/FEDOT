from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Union, Literal

import numpy as np
from golem.core.log import default_log

from fedot.core.data.data import InputData, OutputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.ensembling.utils import (
    calculate_validation_metrics,
)
from fedot.core.pipelines.ensembling.routing import ConstrainedGatingRouter, SamplingRoutingContext
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import TaskTypesEnum


class PipelineEnsemble:
    def __init__(self,
                 pipelines: Sequence[Pipeline],
                 validation_metric: str,
                 ensemble_method: Literal['weighted', 'voting', 'routed_weighted', 'gated_weighted'] = 'voting',
                 pipeline_infos: Optional[List[Dict[str, Any]]] = None,
                 routing_context: Optional[SamplingRoutingContext] = None,
                 ensemble_params: Optional[Dict[str, Any]] = None,
                 batch_size: int = 10000):
        if not pipelines:
            raise ValueError('Pipeline ensemble requires at least one pipeline.')
        self.pipelines: List[Pipeline] = list(pipelines)
        self.log = default_log(self)
        self.ensemble_method = ensemble_method
        self.validation_metric = validation_metric
        self.routing_context = routing_context
        self.ensemble_params = ensemble_params or {}
        self.batch_size = batch_size
        self._is_finalized = False
        self._gating_router: Optional[ConstrainedGatingRouter] = None
        self._gating_partition_names: List[str] = []

        if pipeline_infos is not None:
            self.pipeline_infos = list(pipeline_infos)
            self._sync_pipelines_from_infos()
        else:
            self.pipeline_infos = [
                {
                    'name': f'chunk_{idx}',
                    'pipeline': pipeline,
                    'data_size': None,
                    'metrics': {},
                    'val_predictions': None,
                    'val_probabilities': None,
                }
                for idx, pipeline in enumerate(self.pipelines)
            ]

    @property
    def is_fitted(self) -> bool:
        return all(pipeline.is_fitted for pipeline in self.pipelines)

    def predict(self,
                input_data: Union[InputData, MultiModalData],
                output_mode: str = 'default',
                predictions_cache=None,
                fold_id: Optional[int] = None) -> OutputData:
        if not self.is_fitted:
            raise ValueError('Pipeline ensemble is not fitted yet')

        if isinstance(input_data, InputData) and len(input_data.idx) > self.batch_size:
            prediction = self._predict_ensemble_in_batches(
                input_data=input_data,
                stage='inference',
                output_mode=output_mode,
                predictions_cache=predictions_cache,
                fold_id=fold_id,
            )
        else:
            prediction = self._predict_ensemble(input_data=input_data,
                                                stage='inference',
                                                output_mode=output_mode,
                                                predictions_cache=predictions_cache,
                                                fold_id=fold_id)
        return self._build_output_data(input_data, prediction)

    def _predict_ensemble_in_batches(self,
                                     input_data: InputData,
                                     stage: str,
                                     output_mode: str,
                                     pipeline_infos: Optional[List[Dict[str, Any]]] = None,
                                     predictions_cache=None,
                                     fold_id: Optional[int] = None) -> np.ndarray:
        batch_predictions = []

        for start in range(0, len(input_data.idx), self.batch_size):
            stop = min(start + self.batch_size, len(input_data.idx))
            batch_data = input_data.subset_by_positions(np.arange(start, stop))
            batch_predictions.append(
                self._predict_ensemble(input_data=batch_data,
                                       stage=stage,
                                       output_mode=output_mode,
                                       pipeline_infos=pipeline_infos,
                                       predictions_cache=predictions_cache,
                                       fold_id=fold_id)
            )

        if not batch_predictions:
            return np.array([])

        first = np.asarray(batch_predictions[0])
        if first.ndim == 1:
            return np.concatenate([np.asarray(item).reshape(-1) for item in batch_predictions], axis=0)
        return np.vstack([np.asarray(item) for item in batch_predictions])

    def evaluate_on_data(self,
                         validation_data: InputData,
                         pipeline_infos: Optional[List[Dict[str, Any]]] = None,
                         stage: str = 'validation') -> Dict[str, float]:
        labels, proba = self._predict_ensemble_with_proba(
            input_data=validation_data,
            stage=stage,
            output_mode='default',
            pipeline_infos=pipeline_infos,
        )
        return calculate_validation_metrics(
            y_true=validation_data.target,
            y_labels=labels,
            task_type=validation_data.task.task_type,
            y_proba=proba,
        )

    def select_best_models(self,
                           validation_data: InputData,
                           validation_metric: str) -> tuple[list[int], Optional[float]]:
        if not self.pipeline_infos:
            return [], None

        best_subset: tuple[int, ...] = ()
        best_score: Optional[float] = None
        best_gating_router: Optional[ConstrainedGatingRouter] = None
        best_gating_partition_names: List[str] = []

        pipeline_indices = range(len(self.pipeline_infos))
        for subset_size in range(1, len(self.pipeline_infos) + 1):
            for candidate in combinations(pipeline_indices, subset_size):
                subset_infos = [self.pipeline_infos[idx] for idx in candidate]
                if self.ensemble_method == 'gated_weighted':
                    self._fit_gating_router(validation_data, subset_infos)
                metrics = self.evaluate_on_data(validation_data=validation_data, pipeline_infos=subset_infos)
                score = float(metrics[validation_metric])

                if best_score is None or score < best_score:
                    best_subset = candidate
                    best_score = score
                    best_gating_router = self._gating_router
                    best_gating_partition_names = list(self._gating_partition_names)

        selected_infos = [self.pipeline_infos[idx] for idx in best_subset]
        self.pipeline_infos = selected_infos
        self._sync_pipelines_from_infos()
        if self.ensemble_method == 'gated_weighted':
            self._gating_router = best_gating_router
            self._gating_partition_names = best_gating_partition_names
        self.log.message(
            f'Best ensemble subset: models={len(best_subset)} {validation_metric}={best_score}'
        )
        return list(best_subset), best_score

    def finalize(self, validation_data: Optional[InputData] = None) -> 'PipelineEnsemble':
        self._keep_only_fitted_pipeline_infos()

        if validation_data is not None:
            if self.ensemble_method == 'gated_weighted':
                self._fit_gating_router(validation_data, self.pipeline_infos)
            full_metrics = self.evaluate_on_data(validation_data=validation_data, stage='validation')
            self.log.message(f'Ensemble metrics before subset selection: {full_metrics}')

            if len(self.pipelines) > 1:
                selected, best_score = self.select_best_models(
                    validation_data=validation_data,
                    validation_metric=self.validation_metric,
                )
                reduced_metrics = self.evaluate_on_data(validation_data=validation_data, stage='validation')
                self.log.message(f'Ensemble metrics after subset selection: {reduced_metrics}')
                self.log.message(
                    f'Best validation metric after subset selection ({self.validation_metric}): {best_score}. '
                    f'Selected pipelines: {len(selected)}'
                )

        self._is_finalized = True
        return self

    def _predict_ensemble(self,
                          input_data: Union[InputData, MultiModalData],
                          stage: str,
                          output_mode: str,
                          pipeline_infos: Optional[List[Dict[str, Any]]] = None,
                          predictions_cache=None,
                          fold_id: Optional[int] = None) -> np.ndarray:
        predictions, _ = self._predict_ensemble_with_proba(
            input_data=input_data,
            stage=stage,
            output_mode=output_mode,
            pipeline_infos=pipeline_infos,
            predictions_cache=predictions_cache,
            fold_id=fold_id,
        )
        return predictions

    def _predict_ensemble_with_proba(self,
                                     input_data: Union[InputData, MultiModalData],
                                     stage: str,
                                     output_mode: str,
                                     pipeline_infos: Optional[List[Dict[str, Any]]] = None,
                                     predictions_cache=None,
                                     fold_id: Optional[int] = None) -> tuple[np.ndarray, Optional[np.ndarray]]:
        active_infos = self._resolve_active_infos(pipeline_infos)
        if not active_infos:
            raise ValueError('Pipeline ensemble has no pipelines to aggregate.')

        task_type = input_data.task.task_type

        if task_type == TaskTypesEnum.classification:
            labels_per_pipeline: List[np.ndarray] = []
            proba_per_pipeline: List[np.ndarray] = []
            for info in active_infos:
                labels, proba = self._get_pipeline_predictions(
                    info=info,
                    input_data=input_data,
                    stage=stage,
                    predictions_cache=predictions_cache,
                    fold_id=fold_id,
                )
                labels_per_pipeline.append(labels)
                proba_per_pipeline.append(proba)

            weights = self._compute_aggregation_weights(input_data, active_infos)
            aggregated_labels, aggregated_proba = self._aggregate_classification_predictions(
                labels_per_pipeline=labels_per_pipeline,
                proba_per_pipeline=proba_per_pipeline,
                weights=weights,
                output_mode=output_mode,
            )
            return aggregated_labels, aggregated_proba

        predictions = []
        for info in active_infos:
            labels, _ = self._get_pipeline_predictions(
                info=info,
                input_data=input_data,
                stage=stage,
                predictions_cache=predictions_cache,
                fold_id=fold_id,
            )
            predictions.append(labels.astype(float))

        stacked = np.vstack(predictions)
        if self.ensemble_method == 'weighted':
            weights = self._compute_model_weights(active_infos)
            aggregated = np.average(stacked, axis=0, weights=weights)
        elif self.ensemble_method in ('routed_weighted', 'gated_weighted'):
            weights = self._compute_aggregation_weights(input_data, active_infos)
            aggregated = np.sum(stacked.T * weights, axis=1)
        else:
            aggregated = np.mean(stacked, axis=0)
        return aggregated, None

    def _get_pipeline_predictions(self,
                                  info: Dict[str, Any],
                                  input_data: Union[InputData, MultiModalData],
                                  stage: str,
                                  predictions_cache=None,
                                  fold_id: Optional[int] = None) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if stage == 'validation' and info.get('val_predictions') is not None:
            val_pred = np.ravel(np.asarray(info.get('val_predictions')))
            return val_pred, np.asarray(info.get('val_probabilities'))

        pipeline: Pipeline = info['pipeline']
        output_mode = 'labels' if input_data.task.task_type == TaskTypesEnum.classification else 'default'
        labels_output = pipeline.predict(input_data,
                                         output_mode=output_mode,
                                         predictions_cache=predictions_cache,
                                         fold_id=fold_id)
        labels = np.ravel(np.asarray(labels_output.predict))

        if input_data.task.task_type == TaskTypesEnum.classification:
            proba_output = pipeline.predict(input_data,
                                            output_mode='probs',
                                            predictions_cache=predictions_cache,
                                            fold_id=fold_id)
            return labels, np.asarray(proba_output.predict)

        return labels, None

    def _aggregate_classification_predictions(self,
                                              labels_per_pipeline: List[np.ndarray],
                                              proba_per_pipeline: List[np.ndarray],
                                              weights: np.ndarray,
                                              output_mode: str) -> tuple[np.ndarray, Optional[np.ndarray]]:
        stacked_proba = np.stack([np.asarray(proba) for proba in proba_per_pipeline], axis=0)
        if self.ensemble_method in ('weighted', 'routed_weighted', 'gated_weighted'):
            aggregated_proba = self._aggregate_probability_matrix(stacked_proba, weights)
        else:
            aggregated_proba = np.mean(stacked_proba, axis=0)

        if output_mode in ('probs', 'full_probs'):
            return aggregated_proba, aggregated_proba

        classes = np.unique(np.concatenate([np.ravel(np.asarray(labels)) for labels in labels_per_pipeline]))
        label_indices = np.argmax(aggregated_proba, axis=1)
        labels = classes[label_indices] if len(classes) == aggregated_proba.shape[1] else label_indices
        return labels, aggregated_proba

    def _compute_aggregation_weights(self,
                                     input_data: Union[InputData, MultiModalData],
                                     active_infos: Sequence[Dict[str, Any]]) -> np.ndarray:
        if self.ensemble_method == 'weighted':
            return self._compute_model_weights(active_infos)
        if self.ensemble_method == 'routed_weighted':
            base_weights = self._compute_base_routing_weights(input_data, active_infos)
            validation_weights = self._compute_model_weights(active_infos)
            return self._normalize_weight_rows(base_weights * validation_weights.reshape(1, -1))
        if self.ensemble_method == 'gated_weighted':
            return self._compute_gated_weights(input_data, active_infos)
        return np.ones(len(active_infos), dtype=float) / float(len(active_infos))

    def _compute_model_weights(self, active_infos: Sequence[Dict[str, Any]]) -> np.ndarray:
        raw_scores = []
        for info in active_infos:
            value = info.get('metrics', {}).get(self.validation_metric)
            if value is None or not np.isfinite(value):
                raw_scores.append(np.nan)
            else:
                raw_scores.append(float(value))

        if all(np.isnan(raw_scores)):
            return np.ones(len(active_infos), dtype=float) / float(len(active_infos))

        eps = 1e-12
        scores = np.asarray(raw_scores, dtype=float)
        finite_mask = np.isfinite(scores)
        worst_score = float(np.max(scores[finite_mask]))
        weights = np.zeros(len(scores), dtype=float)
        weights[finite_mask] = worst_score - scores[finite_mask] + eps
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0:
            return np.ones(len(active_infos), dtype=float) / float(len(active_infos))
        return weights / total

    def _compute_base_routing_weights(self,
                                      input_data: Union[InputData, MultiModalData],
                                      active_infos: Sequence[Dict[str, Any]]) -> np.ndarray:
        context = self._require_routing_context()
        return context.base_weights(input_data, self._partition_names(active_infos))

    def _compute_gated_weights(self,
                               input_data: Union[InputData, MultiModalData],
                               active_infos: Sequence[Dict[str, Any]]) -> np.ndarray:
        partition_names = self._partition_names(active_infos)
        if self._gating_router is None or partition_names != self._gating_partition_names:
            raise ValueError('Gated ensemble router is not fitted for the active pipelines.')
        features = self._require_routing_context().transform_features(input_data)
        return self._normalize_weight_rows(self._gating_router.weights(features))

    def _fit_gating_router(self,
                           validation_data: InputData,
                           active_infos: Sequence[Dict[str, Any]]) -> None:
        if validation_data.task.task_type != TaskTypesEnum.regression:
            raise ValueError('Gated ensemble currently supports regression tasks only.')
        routing_context = self._require_routing_context()
        partition_names = self._partition_names(active_infos)
        features = routing_context.transform_features(validation_data)
        prior_weights = routing_context.base_weights(validation_data, partition_names)
        predictions = np.column_stack([
            np.ravel(np.asarray(info.get('val_predictions'))).astype(float)
            for info in active_infos
        ])
        router = ConstrainedGatingRouter(
            hidden_dim=int(self.ensemble_params.get('gating_hidden_dim', 64)),
            epochs=int(self.ensemble_params.get('gating_epochs', 200)),
            lr=float(self.ensemble_params.get('gating_lr', 1e-3)),
            kl_weight=float(self.ensemble_params.get('gating_kl_weight', 0.10)),
            balance_weight=float(self.ensemble_params.get('gating_balance_weight', 0.01)),
            weight_decay=float(self.ensemble_params.get('gating_weight_decay', 1e-4)),
            batch_size=int(self.ensemble_params.get('gating_batch_size', 2048)),
            device=str(self.ensemble_params.get('gating_device', 'auto')),
        ).fit(
            features=features,
            prior_weights=prior_weights,
            predictions=predictions,
            target=validation_data.target,
        )
        self._gating_router = router
        self._gating_partition_names = partition_names

    def _require_routing_context(self) -> SamplingRoutingContext:
        if self.routing_context is None:
            raise ValueError(f'Ensemble method "{self.ensemble_method}" requires sampling routing context.')
        return self.routing_context

    @staticmethod
    def _partition_names(active_infos: Sequence[Dict[str, Any]]) -> List[str]:
        return [str(info.get('name')) for info in active_infos]

    @staticmethod
    def _normalize_weight_rows(weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=float)
        row_sums = weights.sum(axis=1, keepdims=True)
        fallback = np.full_like(weights, 1.0 / max(weights.shape[1], 1), dtype=float)
        return np.divide(weights, row_sums, out=fallback, where=row_sums > 0)

    @staticmethod
    def _aggregate_probability_matrix(stacked_proba: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if weights.ndim == 1:
            return np.average(stacked_proba, axis=0, weights=weights)
        return np.sum(stacked_proba * weights.T[:, :, None], axis=0)

    def _resolve_active_infos(self, pipeline_infos: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if pipeline_infos is not None:
            return list(pipeline_infos)
        if self.pipeline_infos:
            return list(self.pipeline_infos)
        return [{'pipeline': pipeline, 'metrics': {}} for pipeline in self.pipelines]

    def _sync_pipelines_from_infos(self) -> None:
        self.pipelines = [info['pipeline'] for info in self.pipeline_infos]

    def _keep_only_fitted_pipeline_infos(self) -> None:
        fitted_infos = [info for info in self.pipeline_infos if info['pipeline'].is_fitted]
        skipped = len(self.pipeline_infos) - len(fitted_infos)
        if skipped:
            self.log.message(f'Skipped unfitted ensemble pipelines: {skipped}')
        if not fitted_infos:
            raise ValueError('Pipeline ensemble has no fitted pipelines to finalize.')
        self.pipeline_infos = fitted_infos
        self._sync_pipelines_from_infos()

    @staticmethod
    def _build_output_data(input_data: Union[InputData, MultiModalData], prediction: np.ndarray) -> OutputData:
        return OutputData(
            idx=input_data.idx,
            features=getattr(input_data, 'features', None),
            task=input_data.task,
            data_type=input_data.data_type,
            target=input_data.target,
            predict=prediction,
        )
