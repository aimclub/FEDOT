from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Union

import numpy as np
import seaborn as sns

from fedot.core.optimisers.opt_history_objects.opt_history import OptHistory
from fedot.core.repository.operation_types_repository import OperationTypesRepository, get_visualization_tags_map
from fedot.core.visualisation.graph_viz import GraphVisualizer
from fedot.core.visualisation.opt_history.utils import LabelsColorMapType
from fedot.core.visualisation.opt_viz import OptHistoryVisualizer


if TYPE_CHECKING:
    from fedot.core.dag.graph import Graph
    from fedot.core.optimisers.graph import OptGraph

    GraphType = Union[Graph, OptGraph]


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
    def __init__(self, history: OptHistory, visuals_params: Optional[Dict[str, Any]] = None):
        visuals_params = visuals_params if visuals_params is not None else {}
        visuals_params['tags_map'] = visuals_params.get('tags_map') or get_visualization_tags_map()
        visuals_params['palette'] = visuals_params.get('palette') or get_palette_based_on_default_tags()
        visuals_params['graph_show_params'] = (visuals_params.get('graph_show_params') or
                                               get_pipeline_show_default_params())

        super().__init__(history, visuals_params)


class PipelineVisualizer(GraphVisualizer):
    def __init__(self, graph: GraphType, visuals_params: Optional[Dict[str, Any]] = None):
        visuals_params = visuals_params or {}
        default_visuals_params = get_pipeline_show_default_params()
        default_visuals_params.update(visuals_params)
        super().__init__(graph, default_visuals_params)
