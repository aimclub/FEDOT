import os
from copy import deepcopy
from inspect import signature
from typing import Optional

from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.tree._tree import TREE_LEAF

from fedot.core.composer.metrics import Metric
from fedot.core.composer.metrics import R2, F1
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.explainability.explainer_template import Explainer


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
        'dt': {'max_depth': 5},
        'dtreg': {'max_depth': 5},
    }

    def __init__(self, model: 'Pipeline', surrogate: str):
        super().__init__(model)

        self.score: Optional[float] = None

        if not isinstance(surrogate, str):
            raise ValueError(f'{surrogate} is not supported as a surrogate model')
        if surrogate not in self.surrogates_default_params:
            raise ValueError(f'{type(surrogate)} is not supported as a surrogate model')

        self.surrogate_str = surrogate
        self.surrogate = get_simple_pipeline(self.surrogate_str, self.surrogates_default_params[surrogate])

    def explain(self, data: InputData, visualization: bool = False, **kwargs):
        try:
            self.score = fit_naive_surrogate_model(self.model, self.surrogate, data)

        except Exception as ex:
            print(f'Failed to fit the surrogate: {ex}')
            return

        # Pruning redundant branches and leaves
        if self.surrogate_str in ('dt', 'dtreg'):
            prune_duplicate_leaves(self.surrogate.root_node.fitted_operation)

        if visualization:
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


def get_simple_pipeline(model: str, custom_params: dict = None) -> 'Pipeline':
    surrogate_node = PrimaryNode(model)
    if custom_params:
        surrogate_node.parameters = custom_params
    return Pipeline(surrogate_node)


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
    data.target = prediction.predict
    surrogate_model.fit(data)

    data_c = deepcopy(data)
    data_c.target = surrogate_model.predict(data, output_mode=output_mode).predict
    score = round(abs(metric.metric(data_c, prediction)), 2)

    return score


def is_leaf(inner_tree, index):
    # Check whether node is leaf node
    return (inner_tree.children_left[index] == TREE_LEAF and
            inner_tree.children_right[index] == TREE_LEAF)


def prune_index(inner_tree, decisions, index=0):
    # Start pruning from the bottom - if we start from the top, we might miss
    # nodes that become leaves during pruning.
    # Do not use this directly - use prune_duplicate_leaves instead.
    if not is_leaf(inner_tree, inner_tree.children_left[index]):
        prune_index(inner_tree, decisions, inner_tree.children_left[index])
    if not is_leaf(inner_tree, inner_tree.children_right[index]):
        prune_index(inner_tree, decisions, inner_tree.children_right[index])

    # Prune children if both children are leaves now and make the same decision:
    if (is_leaf(inner_tree, inner_tree.children_left[index]) and
            is_leaf(inner_tree, inner_tree.children_right[index]) and
            (decisions[index] == decisions[inner_tree.children_left[index]]) and
            (decisions[index] == decisions[inner_tree.children_right[index]])):
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF


def prune_duplicate_leaves(mdl):
    """
    Function for pruning redundant leaves of a tree by Thomas (https://stackoverflow.com/users/4629950/thomas).
    Source: https://stackoverflow.com/questions/51397109/prune-unnecessary-leaves-in-sklearn-decisiontreeclassifier
    :param mdl: `DecisionTree` or `DecisionTreeRegressor` instance by sklearn.
    """
    # Remove leaves if both
    decisions = mdl.tree_.value.argmax(axis=2).flatten().tolist()  # Decision for each node
    prune_index(mdl.tree_, decisions)
