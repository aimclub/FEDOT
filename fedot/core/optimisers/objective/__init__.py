try:
    from .data_objective_eval import PipelineObjectiveEvaluate, DataSource
    from .metrics_objective import MetricsObjective
    from .objective_serialization import init_backward_serialize_compat

    init_backward_serialize_compat()
except ModuleNotFoundError as ex:
    if ex.name and not ex.name.startswith('golem'):
        raise
    PipelineObjectiveEvaluate = None
    DataSource = None
    MetricsObjective = None
