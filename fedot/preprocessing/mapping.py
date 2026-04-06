from fedot.preprocessing.preprocessor_types import (PreprocessingStepEnum, 
                                                    ImputationMethodEnum, 
                                                    ScalingMethodEnum, 
                                                    EmbeddingMethodEnum,
                                                    EncodingMethodEnum)
from fedot.preprocessing.imputation import (MeanImputation, MedianImputation,
                                            ModeImputation, ConstantImputation,
                                            DeleteRawImputation)
from fedot.preprocessing.embedding import TransformerEmbedder
from fedot.preprocessing.categorical_encoding import LabelEncoder, OneHotEncoder


PREPROCESSING_OPTIONAL_MAPPING = {
    PreprocessingStepEnum.imputation: {
        ImputationMethodEnum.mean: MeanImputation,
        ImputationMethodEnum.median: MedianImputation,
        ImputationMethodEnum.mode: ModeImputation,
        ImputationMethodEnum.constant: ConstantImputation,
        ImputationMethodEnum.delete_raw: DeleteRawImputation
    },
    PreprocessingStepEnum.scaling: {
        # ScalingMethodEnum.min_max: MinMaxScalingHandler(),
        # ScalingMethodEnum.standard: StandardScalingHandler(),
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
