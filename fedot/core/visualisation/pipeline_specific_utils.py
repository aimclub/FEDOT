from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np
import seaborn as sns

from fedot.core.optimisers.opt_history_objects.opt_history import OptHistory
from fedot.core.repository.operation_types_repository import OperationTypesRepository, get_visualization_tags_map
from fedot.core.visualisation.opt_history.utils import LabelsColorMapType
from fedot.core.visualisation.opt_viz import OptHistoryVisualizer


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


class PipelineHistoryVisualizer(OptHistoryVisualizer):
    def __init__(self, history: OptHistory, tags_map=None, palette=None, graph_show_params=None):
        if tags_map is None:
            tags_map = get_visualization_tags_map()
        if palette is None:
            palette = get_palette_based_on_default_tags()
        if graph_show_params is None:
            graph_show_params = get_pipeline_show_default_params()

        super().__init__(history, tags_map, palette, graph_show_params)
