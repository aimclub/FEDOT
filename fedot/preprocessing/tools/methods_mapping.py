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
                                                       RobustScaling)
from fedot.industrial.core.architecture.preprocessing.ts_methods.scaling_normalization import (
    SeasonalNormalization,
    RollingNormalization,
    PerChannelNormalization
)
from fedot.industrial.core.architecture.preprocessing.ts_methods.imputation import (
    TSMeanImputation,
    TSMedianImputation,
    TSConstantImputation,
    TSFillImputation,
    TSRollingImputation,
    TSKalmanImputation,
    TSLinearInterpolation,
    TSPolynomialInterpolation,
    TSSplineInterpolation
)
from fedot.industrial.core.architecture.preprocessing.ts_methods.image_preprocessing import (ContrastEqualization,
                                                                        ContrastStretching,
                                                                        GammaCorrection,
                                                                        LogTransform)
from fedot.preprocessing.methods.filtering import QuantileClipping
from fedot.preprocessing.methods.embedding import TransformerEmbedder
from fedot.preprocessing.methods.categorical_encoding import LabelEncoder, OneHotEncoder


PREPROCESSING_OBLIGATORY_MAPPING = {
    PreprocessingStepEnum.embedding: {
        EmbeddingMethodEnum.transformer: TransformerEmbedder,
    },
    PreprocessingStepEnum.target_encoding: {
        EncodingMethodEnum.label: LabelEncoder
    },
    PreprocessingStepEnum.encoding: {
        EncodingMethodEnum.label: LabelEncoder,
        EncodingMethodEnum.ohe: OneHotEncoder,
    }
}


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
    }
}


TS_PREPROCESSING_MAPPING = {
    PreprocessingStepEnum.scaling: {
        ScalingMethodEnum.seasonal: SeasonalNormalization,
        ScalingMethodEnum.rolling: RollingNormalization,
        ScalingMethodEnum.standart_per_channel: PerChannelNormalization,
    },
    PreprocessingStepEnum.image_preprocessing: {
        ImagePreprocessingMethodEnum.contrast_equalization: ContrastEqualization,
        ImagePreprocessingMethodEnum.contrast_stretching: ContrastStretching,
        ImagePreprocessingMethodEnum.gamma_correction: GammaCorrection,
        ImagePreprocessingMethodEnum.log_transformation: LogTransform
    },
    PreprocessingStepEnum.imputation: {
        ImputationMethodEnum.ts_mean: TSMeanImputation,
        ImputationMethodEnum.ts_median: TSMedianImputation,
        ImputationMethodEnum.ts_constant: TSConstantImputation,
        ImputationMethodEnum.ts_fill: TSFillImputation,
        ImputationMethodEnum.ts_rolling: TSRollingImputation,
        ImputationMethodEnum.ts_kalman: TSKalmanImputation,
        ImputationMethodEnum.ts_linear_inter: TSLinearInterpolation,
        ImputationMethodEnum.ts_polynomial_inter: TSPolynomialInterpolation,
        ImputationMethodEnum.ts_spline_inter: TSSplineInterpolation
    }
}
