from fedot.preprocessing.preprocessor_types import (PreprocessingStepEnum, 
                                                    ImputationMethodEnum, 
                                                    ScalingMethodEnum, 
                                                    EmbeddingMethodEnum,
                                                    EncodingMethodEnum)
from fedot.preprocessing.imputation import SimpleImputationHandler
from fedot.preprocessing.embedding import TransformerEmbedder
from fedot.preprocessing.categorical_encoding import LabelEncoder, OneHotEncoder


PREPROCESSING_OPTIONAL_MAPPING = {
    PreprocessingStepEnum.imputation: {
        ImputationMethodEnum.simple: SimpleImputationHandler(),
        # ImputationMethodEnum.moda: ModaImputationHandler(),
    },
    PreprocessingStepEnum.scaling: {
        # ScalingMethodEnum.min_max: MinMaxScalingHandler(),
        # ScalingMethodEnum.standard: StandardScalingHandler(),
    },
    PreprocessingStepEnum.encoding: {
        ...
    },
}


PREPROCESSING_OBLIGATORY_MAPPING = {
    PreprocessingStepEnum.embedding: {
        EmbeddingMethodEnum.transformer: TransformerEmbedder,
    },
    PreprocessingStepEnum.encoding: {
        EncodingMethodEnum.label: LabelEncoder,
        EncodingMethodEnum.ohe: OneHotEncoder,
    }
}
