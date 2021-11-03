from abc import abstractmethod
from typing import Optional
from inspect import signature

from matplotlib import pyplot as plt
from sklearn import tree

from fedot.core.data.data import InputData
import fedot.core.explainability.utils as utils
from fedot.core.repository.tasks import TaskTypesEnum


class Explainer:
    """
    An abstract class for various explanation methods.
    """
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def explain(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def output(self, *args, **kwargs):
        raise NotImplementedError


class SurrogateExplainer(Explainer):
    """
    Base class used for composite model structure definition

    :param model: Pipeline object to be explained
    :param surrogate: surrogate name. Supported surrogates: [dt, dtreg]

    .. note::
        score stores the score of surrogate's prediction on model (equals None if the 'explain' method hasn't been
        called yet)
    """

    surrogates_default_params = {
        'dt': {'max_depth': 3},
        'dtreg': {'max_depth': 3},
    }

    def __init__(self, model: 'Pipeline', surrogate: str):
        super().__init__(model)

        self.score: Optional[float] = None

        if isinstance(surrogate, str):

            if surrogate in self.surrogates_default_params:
                self.surrogate_str = surrogate
                self.surrogate = \
                    utils.single_node_pipeline(self.surrogate_str, self.surrogates_default_params[surrogate])
            else:
                raise ValueError(f'{surrogate} is not supported as a surrogate model')

        else:
            raise ValueError(f'{type(surrogate)} is not supported as a surrogate model')

    def explain(self, data: InputData, instant_output: bool = True, **kwargs):
        try:
            self.score = utils.fit_naive_surrogate_model(self.model, self.surrogate, data)

        except Exception as ex:
            print(f'Failed to fit the surrogate: {ex}')
            return

        if instant_output:
            self.output(**kwargs)

    def output(self, dpi=300, **kwargs):
        """Print and plot results of the last explanation. Suitable keyword parameters
        are passed to the corresponding plot function.

        :param dpi:The figure DPI, defaults to 100
        :type dpi: int, optional
        """
        plt.figure(dpi=dpi)
        if self.surrogate_str in ['dt', 'dtreg']:

            if self.score is not None:
                print(f'Surrogate\'s model reproduction quality: {self.score}')
            # Plot default parameters
            plot_params = {
                'proportion': True,
                'filled': True,
                'rounded': True,
            }
            # Plot parameters defined by user
            kwargs_params = \
                {par: kwargs[par] for par in kwargs if par in signature(tree.plot_tree).parameters}

            plot_params.update(kwargs_params)

            tree.plot_tree(self.surrogate.root_node.fitted_operation, **plot_params)


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
