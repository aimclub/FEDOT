from enum import Enum, IntEnum


def get_max_tag_value(tags: Enum):
    """ Get max value of Enum and round it to next 1000
        It is used to solve aliasing problem between tags
        when some tags in different enumeration have the same value """
    return (max(tag.value for tag in tags) // 1000 + 1) * 1000


class TagsEnum(IntEnum):
    pass

# TODO test for checking accordance between tags in json and there
# all tags have priority accordance to it value
ExcludedTagsEnum = TagsEnum('ExcludedTagsEnum', ('non_default', ))

ModelTagsEnum = TagsEnum('ModelTagsEnum',
                         ('linear', 'non_linear', 'custom_model', 'tree', 'boosting', 'ts_model', 'deep'),
                         start=get_max_tag_value(ExcludedTagsEnum))

DataOperationTagsEnum = TagsEnum('DataOperationTagsEnum',
                                 ('data_source', 'feature_scaling', 'imputation', 'feature_reduction',
                                  'feature_engineering', 'encoding', 'filtering', 'feature_selection',
                                  'ts_to_table', 'smoothing', 'ts_to_ts', 'text', 'decompose', 'imbalanced',
                                  'data_source_img', 'data_source_text', 'data_source_table',
                                  'data_source_ts', 'feature_space_transformation'),
                                 start=get_max_tag_value(ModelTagsEnum))

ComplexityTags = TagsEnum('TimeTags',
                          ('expensive', 'simple', 'unstable'),
                         start=get_max_tag_value(DataOperationTagsEnum))

OtherTagsEnum = TagsEnum('OtherTagsEnum',
                         ('sklearn', 'ml', 'no_prob', 'new_data_refit', 'neural', 'discriminant',
                          'quadratic', 'time_series', 'correct_params', 'composition', 'custom',
                          'non_lagged', 'bayesian', 'non_multi', 'interpretable', 'nlp', 'nans_ignore',
                          'categorical', 'differential', 'dimensionality_transforming', 'cutting',
                          'non_applicable_for_ts', 'affects_target', 'categorical_ignore', 'automl', 'rapids', 'cuML'),
                         start=get_max_tag_value(ComplexityTags))

PresetsTagsEnum = TagsEnum('PresetsTagsEnum',
                           ('auto', 'best_quality', 'fast_train', 'gpu', 'ts', '*tree', '*automl'),
                           start=get_max_tag_value(OtherTagsEnum))

ALL_TAGS = (ExcludedTagsEnum, ModelTagsEnum, DataOperationTagsEnum, ComplexityTags, OtherTagsEnum, PresetsTagsEnum)
