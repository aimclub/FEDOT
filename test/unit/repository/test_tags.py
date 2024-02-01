import json
from itertools import chain
from typing import List, Any, Hashable

from fedot.core.repository.operation_tags_n_repo_enums import PresetsTagsEnum, ALL_TAGS
from fedot.core.repository.operation_types_repo_enum import REPOSITORY_FOLDER
from fedot.api.api_utils.presets import PresetsEnum


def _extract_keys_recursively(data: dict, key: Hashable) -> List[Any]:
    result = list()
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                result.extend(_extract_keys_recursively(v, key))
            elif k == key:
                result.append(v)
    return result


def test_preset_tags_contains_all_presets():
    preset_tags_str = set(tag.name for tag in PresetsTagsEnum)
    preset_str = set(preset.name.lower() for preset in PresetsEnum)
    assert preset_tags_str > preset_str


def test_tags_contains_all_tags_from_repository():
    all_tags = set(tag.name for tags in ALL_TAGS for tag in tags)
    all_repo_tags = set()
    for repository_file in REPOSITORY_FOLDER.rglob('*.json'):
        with open(repository_file) as file:
            repo = json.load(file)
        tags = set(chain(*_extract_keys_recursively(repo, 'tags')))
        all_repo_tags.update(tags)

    assert all_tags > all_repo_tags


def test_tags_values_do_not_intersect():
    all_tag_values = [tag.value for tags in ALL_TAGS for tag in tags]
    assert len(set(all_tag_values)) == len(all_tag_values)
