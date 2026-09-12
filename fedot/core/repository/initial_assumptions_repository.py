"""Loading helpers for experiment-backed initial pipeline assumptions."""

import json
from pathlib import Path
from typing import Dict, List, Optional

from fedot.core.pipelines.pipeline import Pipeline


INITIAL_ASSUMPTIONS_SCHEMA_VERSION = 1
INITIAL_ASSUMPTIONS_PATH = Path(__file__).with_name('data') / 'initial_assumptions.json'


def load_initial_assumptions_repository(path: Optional[Path] = None) -> Dict:
    """Load and perform inexpensive structural checks on the portfolio JSON."""
    repository_path = Path(path) if path is not None else INITIAL_ASSUMPTIONS_PATH
    with repository_path.open(encoding='utf-8') as repository_file:
        repository = json.load(repository_file)
    _validate_repository(repository)
    return repository


def load_initial_assumption(pipeline_id: str, path: Optional[Path] = None) -> Pipeline:
    """Build one unfitted :class:`Pipeline` from its repository identifier."""
    repository = load_initial_assumptions_repository(path)
    try:
        pipeline_description = repository['pipelines'][pipeline_id]
    except KeyError as ex:
        raise ValueError(f'Unknown initial assumption pipeline: {pipeline_id}') from ex
    return Pipeline().load(pipeline_description['graph'])


def load_initial_assumption_profile(profile_id: str, path: Optional[Path] = None) -> List[Pipeline]:
    """Build all unfitted pipelines referenced by a named selection profile."""
    repository = load_initial_assumptions_repository(path)
    matching_profiles = [
        profile
        for task_profiles in repository['profiles'].values()
        for profile in task_profiles
        if profile['id'] == profile_id
    ]
    if not matching_profiles:
        raise ValueError(f'Unknown initial assumption profile: {profile_id}')
    profile = matching_profiles[0]
    return [load_initial_assumption(pipeline_id, path) for pipeline_id in profile['pipeline_ids']]


def _validate_repository(repository: Dict):
    if repository.get('schema_version') != INITIAL_ASSUMPTIONS_SCHEMA_VERSION:
        raise ValueError('Unsupported initial assumptions repository schema')

    pipelines = repository.get('pipelines')
    profiles = repository.get('profiles')
    if not isinstance(pipelines, dict) or not pipelines:
        raise ValueError('Initial assumptions repository has no pipelines')
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError('Initial assumptions repository has no profiles')

    known_tasks = {'classification', 'regression'}
    profile_ids = set()
    for task_type, task_profiles in profiles.items():
        if task_type not in known_tasks or not isinstance(task_profiles, list):
            raise ValueError(f'Unsupported initial assumption task: {task_type}')
        for profile in task_profiles:
            profile_id = profile.get('id')
            if not profile_id or profile_id in profile_ids:
                raise ValueError(f'Duplicate or missing initial assumption profile id: {profile_id}')
            profile_ids.add(profile_id)
            if not profile.get('pipeline_ids'):
                raise ValueError(f'Initial assumption profile {profile_id} has no pipelines')
            for pipeline_id in profile['pipeline_ids']:
                if pipeline_id not in pipelines:
                    raise ValueError(f'Unknown pipeline {pipeline_id} in profile {profile_id}')
                if pipelines[pipeline_id].get('task_type') != task_type:
                    raise ValueError(f'Pipeline {pipeline_id} has the wrong task type for {profile_id}')

    for pipeline_id, pipeline_description in pipelines.items():
        if pipeline_description.get('task_type') not in known_tasks:
            raise ValueError(f'Pipeline {pipeline_id} has an unsupported task type')
        graph = pipeline_description.get('graph')
        if not isinstance(graph, dict) or not graph.get('nodes'):
            raise ValueError(f'Pipeline {pipeline_id} has no graph nodes')
        operation_ids = [node.get('operation_id') for node in graph['nodes']]
        if len(operation_ids) != len(set(operation_ids)) or 0 not in operation_ids:
            raise ValueError(f'Pipeline {pipeline_id} has invalid operation ids')
        known_operation_ids = set(operation_ids)
        for node in graph['nodes']:
            if not node.get('operation_type'):
                raise ValueError(f'Pipeline {pipeline_id} contains an unnamed operation')
            if not set(node.get('nodes_from', [])).issubset(known_operation_ids):
                raise ValueError(f'Pipeline {pipeline_id} contains an unknown parent operation')
