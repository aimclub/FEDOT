import glob
import os
from pathlib import Path
from typing import Mapping, Iterable, List, Tuple, Union

from fedot.core.utils import fedot_project_root

paths_map = {
    'fedot.utilities.requirements_notificator': 'golem.utilities.requirements_notificator',
    'fedot.utilities.profiler': 'golem.utilities.profiler',
    'fedot.core.optimisers.gp_comp': 'golem.core.optimisers.genetic',
    'fedot.core.log': 'golem.core.log',

    'fedot.golem.core.optimisers.composer_requirements import ComposerRequirements':
        'golem.core.optimisers.optimization_parameters import OptimizationParameters',
    'fedot.core.optimisers.gp_comp.pipeline_composer_requirements':
        'fedot.core.pipelines.pipeline_composer_requirements',

    'fedot.core.adapter': 'golem.core.adapter',
    'fedot.core.dag': 'golem.core.dag',
    'fedot.core.optimisers': 'golem.core.optimisers',
    'fedot.core.utilities': 'golem.core.utilities',

    'fedot.core.serializers': 'golem.serializers',
    'fedot.core.visualisation': 'golem.visualisation',
}

_exceptions = {
    'fedot.core.optimisers.objective.metrics_objective',
    'fedot.core.optimisers.objective.data_objective_advisor',
    'fedot.core.optimisers.objective.data_objective_eval',
    'fedot.core.optimisers.objective.data_source_splitter',
    'fedot.core.visualisation.pipeline_specific_visuals',
    'import PipelineObjectiveEvaluate',
    'import init_backward_serialize_compat',
}


def rename_imports(mapping: Mapping[str, str], contents: Iterable[str]) -> Tuple[int, List[str]]:
    final_lines = []
    edit_count = 0
    for line in contents:
        # filter
        test_line = line.strip()
        if (test_line.startswith('import') or test_line.startswith('from')) and \
                not any(exc in test_line for exc in _exceptions):

            edited_line = line
            for old_, new_ in mapping.items():
                edited_line = edited_line.replace(old_, new_, 1)
            final_lines.append(edited_line)
            if line != edited_line:
                edit_count += 1
        else:
            final_lines.append(line)
    return edit_count, final_lines


def rename_substrings_recursively(mapping: Mapping[str, str],
                                  root_dir: Union[str, Path],
                                  dry_run=False):
    assert os.path.isdir(root_dir)

    for filename in glob.iglob(str(root_dir).rstrip('/') + '/**/*.py', recursive=True):
        with open(filename, 'r') as fpy:
            edit_count, edited_lines = rename_imports(mapping, fpy)
        stat = f'processed {edit_count} lines for: {filename}'
        if dry_run:
            print(stat)
            continue
        if edit_count > 0:
            print(stat)
            with open(filename, 'w') as fpy:
                fpy.writelines(edited_lines)


if __name__ == "__main__":
    """ВАЖНО: Этот скрипт переписывает файлы проекта напрямую, так что 
    перед запуском убедитесь, что все ваши локальные изменения сохранены в git. 
    Тогда в случае чего вы сможете откатиться.

    Для тестового запуска передайте параметр `dry_run=True`.
    Для реального запуска `dry_run=False`.
    """

    fedot_root = fedot_project_root()
    rename_substrings_recursively(paths_map, root_dir=fedot_root, dry_run=True)
