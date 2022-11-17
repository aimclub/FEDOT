import os
from typing import Optional, Union

import seaborn as sns
from matplotlib import pyplot as plt, ticker

from fedot.core.visualisation.opt_history.utils import get_history_dataframe, show_or_save_figure
from fedot.core.visualisation.opt_history.history_visualization import HistoryVisualization


class FitnessBox(HistoryVisualization):
    def visualize(self, save_path: Optional[Union[os.PathLike, str]] = None,
                  dpi: Optional[int] = None, best_fraction: Optional[float] = None):
        """ Visualizes fitness values across generations in the form of boxplot.

        :param save_path: path to save the visualization. If set, then the image will be saved, and if not,
            it will be displayed.
        :param dpi: DPI of the output figure.
        :param best_fraction: fraction of the best individuals of each generation that included in the
            visualization. Must be in the interval (0, 1].
        """
        save_path = save_path or self.get_predefined_value('save_path')
        dpi = dpi or self.get_predefined_value('dpi')
        best_fraction = best_fraction or self.get_predefined_value('best_fraction')

        df_history = get_history_dataframe(self.history, best_fraction=best_fraction)
        columns_needed = ['generation', 'individual', 'fitness']
        df_history = df_history[columns_needed].drop_duplicates(ignore_index=True)
        # Get color palette by mean fitness per generation
        fitness = df_history.groupby('generation')['fitness'].mean()
        fitness = (fitness - min(fitness)) / (max(fitness) - min(fitness))
        colormap = sns.color_palette('YlOrRd', as_cmap=True)

        fig, ax = plt.subplots(figsize=(6.4, 4.8), facecolor='w')
        sns.boxplot(data=df_history, x='generation', y='fitness', palette=fitness.map(colormap), ax=ax)
        fig.set_dpi(dpi)
        fig.set_facecolor('w')

        ax.set_title('Fitness by generations')
        ax.set_xlabel('Generation')
        # Set ticks for every 5 generation if there's more than 10 generations.
        if len(self.history.individuals) > 10:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.xaxis.grid(True)
        str_fraction_of_pipelines = 'all' if best_fraction is None else f'top {best_fraction * 100}% of'
        ax.set_ylabel(f'Fitness of {str_fraction_of_pipelines} generation pipelines')
        ax.yaxis.grid(True)

        show_or_save_figure(fig, save_path, dpi)
