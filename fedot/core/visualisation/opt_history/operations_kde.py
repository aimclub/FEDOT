import os
from typing import Optional, Union

import seaborn as sns
from matplotlib import pyplot as plt

from fedot.core.visualisation.opt_history.history_visualization import HistoryVisualization
from fedot.core.visualisation.opt_history.utils import get_history_dataframe, get_description_of_operations_by_tag, \
    show_or_save_figure, TagOperationsMap, LabelsColorMapType


class OperationsKDE(HistoryVisualization):
    def visualize(self, save_path: Optional[Union[os.PathLike, str]] = None, dpi: Optional[int] = None,
                  best_fraction: Optional[float] = None, tags_map: TagOperationsMap = None,
                  palette: Optional[LabelsColorMapType] = None):
        """ Visualizes operations used across generations in the form of KDE.

        :param save_path: path to save the visualization. If set, then the image will be saved, and if not,
            it will be displayed.
        :param dpi: DPI of the output figure.
        :param best_fraction: fraction of the best individuals of each generation that included in the
            visualization. Must be in the interval (0, 1].
        :param tags_map: if specified, all operations in the history are colored and grouped based on the
            provided tags. If None, operations are not grouped.
        :param palette: a map from operation label to its color. If None, colors are picked by fixed colormap
            for every history independently.
        """

        save_path = save_path or self.get_predefined_value('save_path')
        dpi = dpi or self.get_predefined_value('dpi')
        best_fraction = best_fraction or self.get_predefined_value('best_fraction')
        tags_map = tags_map or self.visualizer.visuals_params.get('tags_map')
        palette = palette or self.visualizer.visuals_params.get('palette')

        generation_column_name = 'Generation'
        operation_column_name = 'Operation'
        column_for_operation = 'tag' if tags_map else 'node'

        df_history = get_history_dataframe(self.history, best_fraction, tags_map)
        df_history = df_history.rename({'generation': generation_column_name,
                                        column_for_operation: operation_column_name}, axis='columns')
        operations_found = df_history[operation_column_name].unique()
        if tags_map:
            tags_all = list(tags_map.keys())
            operations_found = [t for t in tags_all if t in operations_found]  # Sort and filter.

            nodes_per_tag = df_history.groupby(operation_column_name)['node'].unique()
            legend_per_tag = {tag: get_description_of_operations_by_tag(tag, nodes_per_tag[tag], 22)
                              for tag in operations_found}
            df_history[operation_column_name] = df_history[operation_column_name].map(legend_per_tag)
            operations_found = map(legend_per_tag.get, operations_found)
            if palette:
                palette = {legend_per_tag.get(tag): palette.get(tag) for tag in legend_per_tag}

        if not palette:
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

        fig = plot.figure
        fig.set_dpi(dpi)
        fig.set_facecolor('w')
        ax = plt.gca()
        ax.set_xticks(range(len(self.history.individuals)))
        ax.locator_params(nbins=10)
        str_fraction_of_pipelines = 'all' if best_fraction is None else f'top {best_fraction * 100}% of'
        ax.set_ylabel(f'Fraction in {str_fraction_of_pipelines} generation pipelines')

        show_or_save_figure(fig, save_path, dpi)
