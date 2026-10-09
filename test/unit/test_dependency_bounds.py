import re
from collections import Counter
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).parents[2]
PYPROJECT_PATH = PROJECT_ROOT / 'pyproject.toml'
IMMUTABLE_GIT_REFERENCE = re.compile(r'@[0-9a-f]{40}$')


def _load_pyproject():
    with PYPROJECT_PATH.open('rb') as pyproject_file:
        return tomllib.load(pyproject_file)


def _load_requirements(path: Path):
    return [
        line.strip()
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip() and not line.lstrip().startswith('#')
    ]


def _requirement_key(dependency: str):
    requirement = Requirement(dependency)
    return (
        canonicalize_name(requirement.name),
        requirement.url,
        str(requirement.specifier),
        str(requirement.marker),
    )


def _assert_dependency_is_bounded(dependency: str):
    requirement = Requirement(dependency)
    if requirement.url:
        assert IMMUTABLE_GIT_REFERENCE.search(requirement.url), \
            f'{requirement.name} must use an immutable Git commit reference'
        return

    specifiers = list(requirement.specifier)
    assert specifiers, f'{requirement.name} must have version constraints'

    if any(specifier.operator in {'==', '==='} for specifier in specifiers):
        return

    has_lower_bound = any(specifier.operator in {'>', '>=', '~='} for specifier in specifiers)
    has_upper_bound = any(specifier.operator in {'<', '<=', '~='} for specifier in specifiers)
    assert has_lower_bound and has_upper_bound, \
        f'{requirement.name} must have both lower and upper version bounds'


def test_all_dependencies_have_lower_and_upper_bounds():
    pyproject = _load_pyproject()
    dependencies = list(pyproject['build-system']['requires'])
    dependencies.extend(pyproject['project']['dependencies'])
    for optional_dependencies in pyproject['project']['optional-dependencies'].values():
        dependencies.extend(optional_dependencies)

    for dependency in dependencies:
        _assert_dependency_is_bounded(dependency)


@pytest.mark.parametrize(
    ('dependency_group', 'requirements_path'),
    [
        (None, 'requirements.txt'),
        ('docs', 'other_requirements/docs.txt'),
        ('examples', 'other_requirements/examples.txt'),
        ('extra', 'other_requirements/extra.txt'),
        ('profilers', 'other_requirements/profilers.txt'),
    ],
)
def test_pyproject_dependencies_match_requirements_files(dependency_group, requirements_path):
    pyproject = _load_pyproject()
    if dependency_group is None:
        pyproject_dependencies = pyproject['project']['dependencies']
    else:
        pyproject_dependencies = pyproject['project']['optional-dependencies'][dependency_group]

    file_dependencies = _load_requirements(PROJECT_ROOT / requirements_path)
    assert Counter(map(_requirement_key, pyproject_dependencies)) == Counter(map(_requirement_key, file_dependencies))
