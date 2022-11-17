from __future__ import annotations

import os
from pathlib import Path
from textwrap import wrap
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union

import pandas as pd
from matplotlib import pyplot as plt

from fedot.core.log import default_log

if TYPE_CHECKING:
    from fedot.core.optimisers.opt_history_objects.opt_history import OptHistory

MatplotlibColorType = Union[str, Sequence[float]]
LabelsColorMapType = Dict[str, MatplotlibColorType]

TagOperationsMap = Dict[str, List[str]]


def get_history_dataframe(history: OptHistory, best_fraction: Optional[float] = None,
                          tags_map: Optional[TagOperationsMap] = None):
    history_data = {
        'generation': [],
        'individual': [],
        'fitness': [],
        'node': [],
    }
    if tags_map:
        history_data['tag'] = []
        new_map = {}
        for tag, operations in tags_map.items():
            new_map.update({operation: tag for operation in operations})
        tags_map = new_map

    uid_counts = {}  # Resolving individuals with the same uid
    for gen_num, gen in enumerate(history.individuals):
        for ind in gen:
            uid_counts[ind.uid] = uid_counts.get(ind.uid, -1) + 1
            for node in ind.graph.nodes:
                history_data['generation'].append(gen_num)
                history_data['individual'].append('_'.join([ind.uid, str(uid_counts[ind.uid])]))
                fitness = abs(ind.fitness.value)
                history_data['fitness'].append(fitness)
                history_data['node'].append(str(node))
                if not tags_map:
                    continue
                history_data['tag'].append(tags_map.get(str(node), None))

    df_history = pd.DataFrame.from_dict(history_data)

    if best_fraction is not None:
        generation_sizes = df_history.groupby('generation')['individual'].nunique()

        df_individuals = df_history[['generation', 'individual', 'fitness']] \
            .drop_duplicates(ignore_index=True)

        df_individuals['rank_per_generation'] = df_individuals.sort_values('fitness', ascending=False). \
            groupby('generation').cumcount()

        best_individuals = df_individuals[
            df_individuals.apply(
                lambda row: row['rank_per_generation'] < generation_sizes[row['generation']] * best_fraction,
                axis='columns'
            )
        ]['individual']

        df_history = df_history[df_history['individual'].isin(best_individuals)]

    return df_history


def get_description_of_operations_by_tag(tag: str, operations_by_tag: List[str], max_line_length: int,
                                         format_tag: str = 'it'):
    def make_text_fancy(text: str):
        return text.replace('_', ' ')

    def format_text(text_to_wrap: str, latex_format_tag: str = 'it') -> str:
        formatted_text = f'$\\{latex_format_tag}{{{text_to_wrap}}}$'
        formatted_text = formatted_text.replace(' ', '\\;')
        return formatted_text

    def format_wrapped_text_from_beginning(wrapped_text: List[str], part_to_format: str, latex_format_tag: str = 'it') \
            -> List[str]:
        for line_num, line in enumerate(wrapped_text):
            if part_to_format in line:
                # The line contains the whole part_to_format.
                wrapped_text[line_num] = line.replace(part_to_format, format_text(part_to_format, latex_format_tag))
                break
            if part_to_format.startswith(line):
                # The whole line should be formatted.
                wrapped_text[line_num] = format_text(line, latex_format_tag)
                part_to_format = part_to_format[len(line):].strip()

        return wrapped_text

    tag = make_text_fancy(tag)
    operations_by_tag = ', '.join(operations_by_tag)
    description = f'{tag}: {operations_by_tag}.'
    description = make_text_fancy(description)
    description = wrap(description, max_line_length)
    description = format_wrapped_text_from_beginning(description, tag, format_tag)
    description = '\n'.join(description)
    return description


def show_or_save_figure(figure: plt.Figure, save_path: Optional[Union[os.PathLike, str]], dpi: int = 100):
    if not save_path:
        plt.show()
    else:
        save_path = Path(save_path)
        if not save_path.is_absolute():
            save_path = Path.cwd().joinpath(save_path)
        figure.savefig(save_path, dpi=dpi)
        default_log().info(f'The figure was saved to "{save_path}".')
        plt.close()
