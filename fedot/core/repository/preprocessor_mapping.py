from fedot.preprocessing.get_embeddings import encode_text_features
from fedot.preprocessing.preprocessor_types import EmbeddingMethodEnum, EncodingStrategyEnum
from fedot.preprocessing.categorical_encoding import LabelEncoder, OneHotEncoder


EMBEDDING_METHOD_MAPPING = {
    EmbeddingMethodEnum.transformer: encode_text_features,
}


ENCODER_MAPPING = {
    EncodingStrategyEnum.label: LabelEncoder,
    EncodingStrategyEnum.ohe: OneHotEncoder,
}