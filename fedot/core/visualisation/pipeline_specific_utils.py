from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np
import seaborn as sns

from fedot.core.repository.operation_types_repository import OperationTypesRepository, get_visualization_tags_map
from fedot.core.visualisation.opt_history.arg_constraint_wrapper import ArgConstraintWrapper
from fedot.core.visualisation.opt_history.history_visualization import HistoryVisualization
from fedot.core.visualisation.opt_history.utils import LabelsColorMapType


def get_palette_based_on_default_tags() -> LabelsColorMapType:
    default_tags = [*OperationTypesRepository.DEFAULT_MODEL_TAGS, *OperationTypesRepository.DEFAULT_DATA_OPERATION_TAGS]
    p_1 = sns.color_palette('tab20')
    colour_period = 2  # diverge similar nearby colors
    p_1 = [p_1[i // (len(p_1) // colour_period) + i * colour_period % len(p_1)] for i in range(len(p_1))]
    p_2 = sns.color_palette('Set3')
    palette = np.vstack([p_1, p_2])
    palette_map = {tag: palette[i] for i, tag in enumerate(default_tags)}
    palette_map.update({None: 'mediumaquamarine'})
    return palette_map


def get_colors_by_tags(labels: Iterable[str]) -> LabelsColorMapType:
    tags_map = get_visualization_tags_map()
    new_map = {}
    for tag, operations in tags_map.items():
        new_map.update({operation: tag for operation in operations})
    tags_map = new_map

    palette = get_palette_based_on_default_tags()
    return {label: palette[tags_map.get(label)] for label in labels}


def get_pipeline_show_default_params() -> Dict[str, Any]:
    return {
        'node_color': get_colors_by_tags
    }


def set_default_values_for_pipeline():
    def set_tags_map_and_palette_defaults(visualization: HistoryVisualization, **kwargs) -> Dict[str, Any]:
        name = 'tags_map'
        kwargs[name] = kwargs.get(name) or get_visualization_tags_map()
        if 'palette' in kwargs and not kwargs['palette']:
            kwargs['palette'] = get_palette_based_on_default_tags()
        return kwargs

    ArgConstraintWrapper.DEFAULT_CONSTRAINTS['tags_map'] = set_tags_map_and_palette_defaults


set_default_values_for_pipeline()
