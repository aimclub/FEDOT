from fedot.core.repository.tasks import TaskTypesEnum
from fedot.explainability.surrogate_explainer import SurrogateExplainer


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


def explain_pipeline(pipeline: 'Pipeline', data: 'InputData', method: str = 'surrogate_dt',
                     visualization: bool = False, **kwargs) -> 'Explainer':
    """Create explanation for the `pipeline` according to the selected `method`.
    An `Explainer` instance is returned.

    :param pipeline: pipeline to explain.
    :param data: samples to be explained.
    :param method: explanation method, defaults to 'surrogate_dt'. Options: ['surrogate_dt', ...]
    :param visualization: print and plot the explanation simultaneously, defaults to True.
        The explanation can be retrieved later by executing `explainer.visualize()`.
    """
    if not pipeline:
        raise AssertionError('The pipeline might be fit before explanation!')

    explainer = pick_pipeline_explainer(pipeline, method, data.task.task_type)
    explainer.explain(data, visualization=visualization, **kwargs)

    return explainer
