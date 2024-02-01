from fedot.core.repository.operation_tags_n_repo_enums import PresetsTagsEnum
from fedot.api.api_utils.presets import PresetsEnum


def test_preset_tags_contains_all_presets():
    preset_tags_str = set(tag.name for tag in PresetsTagsEnum)
    preset_str = set(preset.name.lower() for preset in PresetsEnum)
    assert preset_tags_str > preset_str
