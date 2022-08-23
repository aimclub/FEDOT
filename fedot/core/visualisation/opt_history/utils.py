from __future__ import annotations

import os
from pathlib import Path
from textwrap import wrap
from typing import Optional, Union, List, Dict, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from fedot.core.log import default_log
from fedot.core.repository.operation_types_repository import get_opt_node_tag, OperationTypesRepository

if TYPE_CHECKING:
    from fedot.core.optimisers.opt_history import OptHistory


def show_or_save_figure(figure: plt.Figure, save_path: Optional[Union[os.PathLike, str]], dpi: int = 300):
    if not save_path:
        plt.show()
    else:
        save_path = Path(save_path)
        if not save_path.is_absolute():
            save_path = Path.cwd().joinpath(save_path)
        figure.savefig(save_path, dpi=dpi)
        default_log().info(f'The figure was saved to "{save_path}".')
        plt.close()


def get_history_dataframe(history: OptHistory, tags_model: Optional[List[str]] = None,
                          tags_data: Optional[List[str]] = None, best_fraction: Optional[float] = None,
                          get_tags: bool = True):
    history_data = {
        'generation': [],
        'individual': [],
        'fitness': [],
        'node': [],
    }
    if get_tags:
        history_data['tag'] = []

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
                if not get_tags:
                    continue
                history_data['tag'].append(get_opt_node_tag(str(node), tags_model=tags_model, tags_data=tags_data))

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


def get_description_of_operations_by_tag(tag: str, operations_by_tag: List[str], max_line_length: int = 22,
                                         format_tag: str = 'it'):
    def make_text_fancy(text: str):
        return text.replace('_', ' ')

    def format_text(text_to_wrap: str, latex_format_tag: str = 'it') -> str:
        formatted_text = '$\\' + latex_format_tag + '{' + text_to_wrap + '}$'
        formatted_text = formatted_text.replace(' ', '\\;')
        return formatted_text

    def format_wrapped_text(wrapped_text: List[str], part_to_format: str, latex_format_tag: str = 'it') -> List[str]:

        long_text = ''.join(wrapped_text)
        first_tag_pos = long_text.find(part_to_format)
        second_tag_pos = first_tag_pos + len(part_to_format)

        line_len = len(wrapped_text[0])

        first_tag_line = first_tag_pos // line_len
        first_tag_char = first_tag_pos % line_len

        second_tag_line = second_tag_pos // line_len
        second_tag_char = second_tag_pos % line_len

        if first_tag_line == second_tag_line:
            wrapped_text[first_tag_line] = (
                    wrapped_text[first_tag_line][:first_tag_char] +
                    format_text(wrapped_text[first_tag_line][first_tag_char:second_tag_char], latex_format_tag) +
                    wrapped_text[first_tag_line][second_tag_char:]
            )
        else:
            for line in range(first_tag_line + 1, second_tag_line):
                wrapped_text[line] = format_text(wrapped_text[line], latex_format_tag)

            wrapped_text[first_tag_line] = (
                wrapped_text[first_tag_line][:first_tag_char] +
                format_text(wrapped_text[first_tag_line][first_tag_char:], latex_format_tag)
            )
            wrapped_text[second_tag_line] = (
                    format_text(wrapped_text[second_tag_line][:second_tag_char], latex_format_tag) +
                wrapped_text[second_tag_line][second_tag_char:]
            )
        return wrapped_text

    tag = make_text_fancy(tag)
    operations_by_tag = ', '.join(operations_by_tag)
    description = f'{tag}: {operations_by_tag}.'
    description = make_text_fancy(description)
    description = wrap(description, max_line_length)
    description = format_wrapped_text(description, tag, format_tag)
    description = '\n'.join(description)
    return description


def get_palette_based_on_default_tags() -> Dict[str, Tuple[float, float, float]]:
    default_tags = [*OperationTypesRepository.DEFAULT_MODEL_TAGS, *OperationTypesRepository.DEFAULT_DATA_OPERATION_TAGS]
    p_1 = sns.color_palette('tab20')
    colour_period = 2  # diverge similar nearby colors
    p_1 = [p_1[i // (len(p_1) // colour_period) + i * colour_period % len(p_1)] for i in range(len(p_1))]
    p_2 = sns.color_palette('Set3')
    palette = np.vstack([p_1, p_2])
    palette_map = {tag: palette[i] for i, tag in enumerate(default_tags)}
    palette_map.update({None: 'mediumaquamarine'})
    return palette_map
