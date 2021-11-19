import os
from typing import Optional
from copy import deepcopy

from inspect import signature
from matplotlib import pyplot as plt
from sklearn import tree

import fedot.core.pipelines.pipeline as pipeline
from fedot.core.explainability.explainer_template import Explainer
from fedot.core.composer.metrics import R2, F1
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.repository.tasks import TaskTypesEnum


class SurrogateExplainer(Explainer):
    """
    Base class used for composite model structure definition

    :param model: `Pipeline` object to be explained
    :param surrogate: surrogate name. Supported surrogates: `[dt, dtreg]`

    .. note::
        `score` stores the score of surrogate's prediction on model (equals None if the 'explain' method hasn't been
        called yet)
    """

    surrogates_default_params = {
        'dt': {'max_depth': 5, 'ccp_alpha': 5*10**-4},
        'dtreg': {'max_depth': 5, 'ccp_alpha': 5*10**-4},
    }

    def __init__(self, model: 'Pipeline', surrogate: str):
        super().__init__(model)

        self.score: Optional[float] = None

        if not isinstance(surrogate, str):
            raise ValueError(f'{surrogate} is not supported as a surrogate model')
        if surrogate not in self.surrogates_default_params:
            raise ValueError(f'{type(surrogate)} is not supported as a surrogate model')

        self.surrogate_str = surrogate
        self.surrogate = simple_pipeline(self.surrogate_str, self.surrogates_default_params[surrogate])

    def explain(self, data: InputData, visualize: bool = True, **kwargs):
        try:
            self.score = fit_naive_surrogate_model(self.model, self.surrogate, data)

        except Exception as ex:
            print(f'Failed to fit the surrogate: {ex}')
            return

        if visualize:
            self.visualize(**kwargs)

    def visualize(self, dpi: int = 100, figsize=(48, 12), save_path: str = None, **kwargs):
        """Print and plot results of the last explanation. Suitable keyword parameters
        are passed to the corresponding plot function.
        :param dpi: the figure DPI, defaults to 100.
        :param figsize: the figure size in format `(width, height)`, defaults to `(48, 12)`.
        :param save_path: path to save the plot.
        """
        plt.figure(dpi=dpi, figsize=figsize)
        if self.surrogate_str in ['dt', 'dtreg']:

            if self.score is not None:
                print(f'Surrogate\'s model reproduction quality: {self.score}')
            # Plot default parameters
            plot_params = {
                'proportion': True,
                'filled': True,
                'rounded': True,
                'fontsize': 12,
            }
            # Plot parameters defined by user
            kwargs_params = \
                {par: kwargs[par] for par in kwargs if par in signature(tree.plot_tree).parameters}

            plot_params.update(kwargs_params)

            tree.plot_tree(self.surrogate.root_node.fitted_operation, **plot_params)

        if save_path is not None:
            plt.savefig(save_path)
            print(f'Saved the plot to "{os.path.abspath(save_path)}"')


def simple_pipeline(model: str, custom_params: dict = None) -> 'Pipeline':
    surrogate_node = PrimaryNode(model)
    if custom_params:
        surrogate_node.custom_params = custom_params
    return pipeline.Pipeline(surrogate_node)


def fit_naive_surrogate_model(
        black_box_model: 'Pipeline', surrogate_model: 'Pipeline', data: 'InputData',
        metric: 'Metric' = None) -> Optional[float]:

    output_mode = 'default'

    if data.task.task_type == TaskTypesEnum.classification:
        output_mode = 'labels'
        if metric is None:
            metric = F1
    elif data.task.task_type == TaskTypesEnum.regression and metric is None:
        metric = R2

    prediction = black_box_model.predict(data, output_mode=output_mode)
    surrogate_model.fit(data, prediction)

    data_c = deepcopy(data)
    data_c.target = surrogate_model.predict(data, output_mode=output_mode).predict
    score = round(abs(metric.metric(data_c, prediction)), 2)

    return score
