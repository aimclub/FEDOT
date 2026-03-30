


class SimpleImputationHandler:
    def fit(self, data):
        ...
        fitted_imputer, updated_features = ...
        return fitted_imputer, updated_features

    def transform(self, data, step: PreprocessingStep, fitted_obj):
        updated_features = transform_simple_imputer(
            features=data.features,
            features_idx=step.features_idx,
            fitted_imputer=fitted_obj,
        )
        return updated_features