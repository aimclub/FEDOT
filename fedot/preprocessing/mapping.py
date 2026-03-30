from fedot.preprocessing.preprocessor_types import PreprocessingStepEnum
from fedot.preprocessing.imputation import SimpleImputationHandler

PREPROCESSING_MAPPING = {
    PreprocessingStepEnum.imputation: {
        ImputationMethodEnum.simple: SimpleImputationHandler(),
        ImputationMethodEnum.moda: ModaImputationHandler(),
    },
    PreprocessingStepEnum.scaling: {
        ScalingMethodEnum.min_max: MinMaxScalingHandler(),
        ScalingMethodEnum.standard: StandardScalingHandler(),
    },
    PreprocessingStepEnum.encoding: {
        ...
    },
}