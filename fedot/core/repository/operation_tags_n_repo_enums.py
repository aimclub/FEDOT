from enum import IntEnum


class TagsEnum(IntEnum):
    pass

# TODO test for checking accordance between tags in json and there
# all tags have priority accordance to it value
ExcludedTagsEnum = TagsEnum('ExcludedTagsEnum', ('non_default', ))
ModelTagsEnum = TagsEnum('ModelTagsEnum', ('linear', 'non_linear', 'custom_model', 'tree', 'boosting', 'ts_model', 'deep'))
DataOperationTagsEnum = TagsEnum('DataOperationTagsEnum', ('data_source', 'feature_scaling', 'imputation', 'feature_reduction',
                                                           'feature_engineering', 'encoding', 'filtering', 'feature_selection',
                                                           'ts_to_table', 'smoothing', 'ts_to_ts', 'text', 'decompose', 'imbalanced',
                                                           'data_source_img', 'data_source_text', 'data_source_table', 'data_source_ts',
                                                           'feature_space_transformation'))
ComplexityTags = TagsEnum('TimeTags', ('expensive', 'simple', 'unstable'))
OtherTagsEnum = TagsEnum('OtherTagsEnum', ('sklearn', 'ml', 'no_prob', 'new_data_refit', 'neural', 'discriminant', 'quadratic',
                                           'time_series', 'correct_params', 'composition', 'custom',
                                           'non_lagged', 'bayesian', 'non_multi', 'interpretable', 'nlp', 'nans_ignore',
                                           'categorical', 'differential', 'dimensionality_transforming', 'cutting',
                                           'non_applicable_for_ts', 'affects_target', 'categorical_ignore', 'automl', 'rapids', 'cuML'))
PresetsTagsEnum = TagsEnum('PresetsTagsEnum', ('auto', 'best_quality', 'fast_train', 'gpu', 'ts', '*tree', '*automl'))

ALL_TAGS = (ExcludedTagsEnum, ModelTagsEnum, DataOperationTagsEnum, ComplexityTags, OtherTagsEnum, PresetsTagsEnum)
