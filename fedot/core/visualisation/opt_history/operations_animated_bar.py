import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import seaborn as sns
from matplotlib import cm, animation, pyplot as plt
from matplotlib.colors import Normalize

from fedot.core.visualisation.opt_history.history_visualization import HistoryVisualization
from fedot.core.visualisation.opt_history.utils import get_history_dataframe, get_description_of_operations_by_tag, \
    TagOperationsMap, LabelsColorMapType


class OperationsAnimatedBar(HistoryVisualization):
    def visualize(self, save_path: Optional[Union[os.PathLike, str]] = None, dpi: Optional[int] = None,
                  best_fraction: Optional[float] = None, show_fitness: Optional[bool] = None,
                  tags_map: TagOperationsMap = None, palette: Optional[LabelsColorMapType] = None):
        """ Visualizes operations used across generations in the form of animated bar plot.

        :param save_path: path to save the visualization.
        :param dpi: DPI of the output figure.
        :param best_fraction: fraction of the best individuals of each generation that included in the
            visualization. Must be in the interval (0, 1].
        :param show_fitness: if False, the bar colors will not correspond to fitness.
        :param tags_map: if specified, all operations in the history are grouped based on the provided tags.
            If None, operations are not grouped.
        :param palette: a map from operation label to its color. If None, colors are picked by fixed colormap
            for every history independently.

        """

        save_path = save_path or self.get_predefined_value('save_path') or 'history_animated_bars.gif'
        dpi = dpi or self.get_predefined_value('dpi')
        best_fraction = best_fraction or self.get_predefined_value('best_fraction')
        show_fitness = show_fitness if show_fitness is not None else self.get_predefined_value('show_fitness') or True
        tags_map = tags_map or self.visualizer.visuals_params.get('tags_map')
        palette = palette or self.visualizer.visuals_params.get('palette')

        def interpolate_points(point_1, point_2, smoothness=18, power=4) -> List[np.array]:
            t_interp = np.linspace(0, 1, smoothness)
            point_1, point_2 = np.array(point_1), np.array(point_2)
            return [point_1 * (1 - t ** power) + point_2 * t ** power for t in t_interp]

        def smoothen_frames_data(data: Sequence[Sequence['ArrayLike']], smoothness=18, power=4) -> List[np.array]:
            final_frames = []
            for initial_frame in range(len(data) - 1):
                final_frames += interpolate_points(data[initial_frame], data[initial_frame + 1], smoothness, power)
            # final frame interpolates into itcls
            final_frames += interpolate_points(data[-1], data[-1], smoothness, power)

            return final_frames

        def animate(frame_num):
            frame_count = bar_data[frame_num]
            frame_color = bar_color[frame_num] if show_fitness else None
            frame_title = bar_title[frame_num]

            plt.title(frame_title)
            for bar_num in range(len(bars)):
                bars[bar_num].set_width(frame_count[bar_num])
                if not show_fitness:
                    continue
                bars[bar_num].set_facecolor(frame_color[bar_num])

        save_path = Path(save_path)
        if save_path.suffix not in ['.gif', '.mp4']:
            raise ValueError('A correct file extension (".mp4" or ".gif") should be set to save the animation.')

        animation_frames_per_step = 18
        animation_interval_between_frames_ms = 40
        animation_interpolation_power = 4
        fitness_colormap = cm.get_cmap('YlOrRd')

        generation_column_name = 'Generation'
        fitness_column_name = 'Fitness'
        operation_column_name = 'Operation'
        column_for_operation = 'tag' if tags_map else 'node'

        df_history = get_history_dataframe(self.history, best_fraction, tags_map)
        df_history = df_history.rename({
            'generation': generation_column_name,
            'fitness': fitness_column_name,
            column_for_operation: operation_column_name,
        }, axis='columns')
        operations_found = df_history[operation_column_name].unique()
        if tags_map:
            tags_all = list(tags_map.keys())
            operations_found = [tag for tag in tags_all if tag in operations_found]
            nodes_per_tag = df_history.groupby(operation_column_name)['node'].unique()
            bars_labels = [get_description_of_operations_by_tag(t, nodes_per_tag[t], 22) for t in operations_found]
        else:
            bars_labels = operations_found

        if palette:
            no_fitness_palette = palette
        else:
            no_fitness_palette = sns.color_palette('tab10', n_colors=len(operations_found))
            no_fitness_palette = {o: no_fitness_palette[i] for i, o in enumerate(operations_found)}

        # Getting normed fraction of individuals  per generation that contain operations given.
        generation_sizes = df_history.groupby(generation_column_name)['individual'].nunique()
        operations_with_individuals_count = df_history.groupby(
            [generation_column_name, operation_column_name],
            as_index=False
        ).aggregate({'individual': 'nunique'})
        operations_with_individuals_count['individual'] = operations_with_individuals_count.apply(
            lambda row: row['individual'] / generation_sizes[row[generation_column_name]],
            axis='columns')

        if show_fitness:
            # Getting fitness per individual with the list of contained operations in the form of
            # '.operation_1.operation_2. ... .operation_n.'
            individuals_fitness = df_history[[generation_column_name, 'individual', fitness_column_name]] \
                .drop_duplicates()
            individuals_fitness['operations'] = individuals_fitness.apply(
                lambda row: '.{}.'.format('.'.join(
                    df_history[
                        (df_history[generation_column_name] == row[generation_column_name]) &
                        (df_history['individual'] == row['individual'])
                        ][operation_column_name])),
                axis='columns')
            # Getting mean fitness of individuals with the operations given.
            operations_with_individuals_count[fitness_column_name] = operations_with_individuals_count.apply(
                lambda row: individuals_fitness[
                    (individuals_fitness[generation_column_name] == row[generation_column_name]) &
                    (individuals_fitness['operations'].str.contains(f'.{row[operation_column_name]}.'))
                    ][fitness_column_name].mean(),
                axis='columns')
            del individuals_fitness
        # Replacing the initial DataFrame with the processed one
        df_history = operations_with_individuals_count.set_index([generation_column_name, operation_column_name])
        del operations_with_individuals_count

        min_fitness = df_history[fitness_column_name].min() if show_fitness else None
        max_fitness = df_history[fitness_column_name].max() if show_fitness else None

        generations = generation_sizes.index.unique()
        bar_data = []
        bar_color = []
        # Getting data by tags through all generations and filling with zeroes where no such tag
        for gen_num in generations:
            bar_data.append([df_history.loc[gen_num]['individual'].get(tag, 0) for tag in operations_found])
            if not show_fitness:
                continue
            fitnesses = [df_history.loc[gen_num][fitness_column_name].get(tag, 0) for tag in operations_found]
            # Transfer fitness to color
            bar_color.append([
                fitness_colormap((fitness - min_fitness) / (max_fitness - min_fitness)) for fitness in fitnesses])

        bar_data = smoothen_frames_data(bar_data, animation_frames_per_step, animation_interpolation_power)
        title_template = 'Generation {}'
        if best_fraction is not None:
            title_template += f', top {best_fraction * 100}%'
        bar_title = [i for gen_num in generations for i in [title_template.format(gen_num)] * animation_frames_per_step]

        fig, ax = plt.subplots(figsize=(8, 5), facecolor='w')
        if show_fitness:
            bar_color = smoothen_frames_data(bar_color, animation_frames_per_step, animation_interpolation_power)
            sm = cm.ScalarMappable(norm=Normalize(min_fitness, max_fitness), cmap=fitness_colormap)
            sm.set_array([])
            fig.colorbar(sm, label=fitness_column_name, ax=ax)

        count = bar_data[0]
        color = bar_color[0] if show_fitness else [no_fitness_palette[tag] for tag in operations_found]
        title = bar_title[0]

        label_size = 10
        if any(len(label.split('\n')) > 2 for label in bars_labels):
            label_size = 8

        bars = ax.barh(bars_labels, count, color=color)
        ax.tick_params(axis='y', which='major', labelsize=label_size)
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_xlabel(f'Fraction of pipelines containing the operation')
        ax.xaxis.grid(True)
        ax.set_ylabel(operation_column_name)
        ax.invert_yaxis()
        plt.tight_layout()

        if not save_path.is_absolute():
            save_path = Path.cwd().joinpath(save_path)

        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=len(bar_data),
            interval=animation_interval_between_frames_ms,
            repeat=True
        )
        ani.save(str(save_path), dpi=dpi)
        self.log.info(f'The animation was saved to "{save_path}".')
        plt.close(fig=fig)
