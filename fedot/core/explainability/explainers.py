from fedot.core.explainability.surrogate_explainer import SurrogateExplainer
from fedot.core.repository.tasks import TaskTypesEnum


def pick_pipeline_explainer(pipeline: 'Pipeline', method: str, task_type: TaskTypesEnum):
    if method == 'surrogate_dt':
        if task_type == TaskTypesEnum.classification:
            surrogate = 'dt'
        elif task_type == TaskTypesEnum.regression:
            surrogate = 'dtreg'
        else:
            raise ValueError(f'Surrogate tree is not applicable for the {task_type} task')
        explainer = SurrogateExplainer(pipeline, surrogate=surrogate)

    else:
        raise ValueError(f'Explanation method {method} is not supported')

    return explainer
