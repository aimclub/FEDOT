from typing import Any, Type

from marshmallow import INCLUDE, Schema, ValidationError, fields, validates_schema

from fedot.core.repository.metrics_repository import MetricsRepository
from fedot.validation.boundaries import load_validated
from fedot.validation.context import ValidationContext

try:
    from golem.core.tuning.iopt_tuner import IOptTuner
except ModuleNotFoundError:
    IOptTuner = None

from golem.core.tuning.optuna_tuner import OptunaTuner


class RegisteredMetricSchema(Schema):
    class Meta:
        unknown = INCLUDE

    metric = fields.Raw(required=True)

    @validates_schema
    def validate_metric(self, data: dict, **kwargs) -> None:
        metric = data['metric']
        if callable(metric):
            return
        if metric not in MetricsRepository._metrics_implementations:
            raise ValidationError(f'Incorrect metric {metric}')


def validate_registered_metric(metric: Any, context: ValidationContext = None) -> None:
    load_validated(
        RegisteredMetricSchema(),
        {'metric': metric},
        context,
        prefix='metrics_objective',
    )


class MultiObjectiveTunerSchema(Schema):
    class Meta:
        unknown = INCLUDE

    metrics_count = fields.Int(required=True)
    tuner_class = fields.Raw(required=True)

    @validates_schema
    def validate_multi_objective_tuner(self, data: dict, **kwargs) -> None:
        if data['metrics_count'] <= 1:
            return
        supported_tuners = {OptunaTuner}
        if IOptTuner is not None:
            supported_tuners.add(IOptTuner)
        if data['tuner_class'] not in supported_tuners:
            raise ValidationError(
                'Multi objective tuning applicable only for OptunaTuner and IOptTuner.')


def validate_multi_objective_tuner(
    tuner_class: Type,
    metrics_count: int,
    context: ValidationContext = None,
) -> None:
    load_validated(
        MultiObjectiveTunerSchema(),
        {
            'metrics_count': metrics_count,
            'tuner_class': tuner_class,
        },
        context,
        prefix='tuner',
    )
