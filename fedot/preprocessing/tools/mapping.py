from fedot.preprocessing.tools.preprocessor_types import (PreprocessingStepEnum, 
                                                    ImputationMethodEnum, 
                                                    ScalingMethodEnum, 
                                                    EmbeddingMethodEnum,
                                                    EncodingMethodEnum,
                                                    FilteringMethodEnum,
                                                    ImagePreprocessingMethodEnum)
from fedot.preprocessing.methods.imputation import (MeanImputation, MedianImputation,
                                            ModeImputation, ConstantImputation,
                                            DeleteRawImputation)
from fedot.preprocessing.methods.scaling_normalization import (StandartScaling, 
                                                       MinMaxNormalization,
                                                       RobustScaling,
                                                       SeasonalNormalization,
                                                       RollingNormalization,
                                                       PerChannelNormalization)
from fedot.preprocessing.multi_channel_methods.image_preprocessing import (ContrastEqualization,
                                                                        ContrastStretching,
                                                                        GammaCorrection,
                                                                        LogTransform)
from fedot.preprocessing.methods.filtering import QuantileClipping
from fedot.preprocessing.methods.embedding import TransformerEmbedder
from fedot.preprocessing.methods.categorical_encoding import LabelEncoder, OneHotEncoder


PREPROCESSING_OPTIONAL_MAPPING = {
    PreprocessingStepEnum.imputation: {
        ImputationMethodEnum.mean: MeanImputation,
        ImputationMethodEnum.median: MedianImputation,
        ImputationMethodEnum.mode: ModeImputation,
        ImputationMethodEnum.constant: ConstantImputation,
        ImputationMethodEnum.delete_raw: DeleteRawImputation
    },
    PreprocessingStepEnum.scaling: {
        ScalingMethodEnum.min_max: MinMaxNormalization,
        ScalingMethodEnum.standard: StandartScaling,
        ScalingMethodEnum.robust: RobustScaling,
        ScalingMethodEnum.seasonal: SeasonalNormalization,
        ScalingMethodEnum.rolling: RollingNormalization,
        ScalingMethodEnum.standart_per_channel: PerChannelNormalization,
    },
    PreprocessingStepEnum.filtering: {
        FilteringMethodEnum.quantile: QuantileClipping,
    },
    PreprocessingStepEnum.image_preprocessing: {
        ImagePreprocessingMethodEnum.contrast_equalization: ContrastEqualization,
        ImagePreprocessingMethodEnum.contrast_stretching: ContrastStretching,
        ImagePreprocessingMethodEnum.gamma_correction: GammaCorrection,
        ImagePreprocessingMethodEnum.log_transformation: LogTransform
    }
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

