from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional, Union

import seaborn as sns
from matplotlib import pyplot as plt

from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.visualisation.opt_history.utils import get_history_dataframe, get_description_of_operations_by_tag, \
    get_palette_based_on_default_tags, show_or_save_figure

if TYPE_CHECKING:
    from fedot.core.optimisers.opt_history import OptHistory


def visualize_operations_kde(history: OptHistory, save_path: Optional[Union[os.PathLike, str]] = None,
                             dpi: int = 300, best_fraction: Optional[float] = None, use_tags: bool = True,
                             tags_model: Optional[List[str]] = None, tags_data: Optional[List[str]] = None):
    """ Visualizes operations used across generations in the form of KDE.

    :param history: OptHistory.
    :param save_path: path to save the visualization. If set, then the image will be saved,
        and if not, it will be displayed.
    :param dpi: DPI of the output figure.
    :param best_fraction: fraction of the best individuals of each generation that included in the visualization.
        Must be in the interval (0, 1].
    :param use_tags: if True (default), all operations in the history are colored and grouped based on FEDOT
        repo tags. If False, operations are not grouped, colors are picked by fixed colormap for every history
        independently.
    :param tags_model: tags for OperationTypesRepository('model') to map the history operations.
        The later the tag, the higher its priority in case of intersection.
    :param tags_data: tags for OperationTypesRepository('data_operation') to map the history operations.
        The later the tag, the higher its priority in case of intersection.
    """

    tags_model = tags_model or OperationTypesRepository.DEFAULT_MODEL_TAGS
    tags_data = tags_data or OperationTypesRepository.DEFAULT_DATA_OPERATION_TAGS

    tags_all = [*tags_model, *tags_data]

    generation_column_name = 'Generation'
    operation_column_name = 'Operation'
    column_for_operation = 'tag' if use_tags else 'node'

    df_history = get_history_dataframe(history, tags_model, tags_data, best_fraction, use_tags)
    df_history = df_history.rename({'generation': generation_column_name,
                                    column_for_operation: operation_column_name}, axis='columns')
    operations_found = df_history[operation_column_name].unique()
    if use_tags:
        operations_found = [t for t in tags_all if t in operations_found]
        nodes_per_tag = df_history.groupby(operation_column_name)['node'].unique()
        legend = [get_description_of_operations_by_tag(tag, nodes_per_tag[tag]) for tag in operations_found]
        palette = get_palette_based_on_default_tags()
    else:
        legend = operations_found
        palette = sns.color_palette('tab10', n_colors=len(operations_found))

    plot = sns.displot(
        data=df_history,
        x=generation_column_name,
        hue=operation_column_name,
        hue_order=operations_found,
        kind='kde',
        clip=(0, max(df_history[generation_column_name])),
        multiple='fill',
        palette=palette
    )

    for text, new_text in zip(plot.legend.texts, legend):
        text.set_text(new_text)

    fig = plot.figure
    fig.set_dpi(dpi)
    fig.set_facecolor('w')
    ax = plt.gca()
    str_fraction_of_pipelines = 'all' if best_fraction is None else f'top {best_fraction * 100}% of'
    ax.set_ylabel(f'Fraction in {str_fraction_of_pipelines} generation pipelines')

    show_or_save_figure(fig, save_path, dpi)