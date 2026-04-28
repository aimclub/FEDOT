from fedot.preprocessing.categorical_encoding import LabelEncoder, OneHotEncoder
from fedot.preprocessing.get_embeddings import encode_text_features
from fedot.preprocessing.preprocessor_types import EmbeddingMethodEnum, EncodingStrategyEnum


"""
Mapping from :class:`EmbeddingMethodEnum` to the embedding function.

The embedding function is expected to take text features and the resolved embedding
parameters and return embeddings (typically as a torch tensor).
"""
EMBEDDING_METHOD_MAPPING = {
    EmbeddingMethodEnum.transformer: encode_text_features,
}


"""
Mapping from :class:`EncodingStrategyEnum` to an encoder class.

The encoder class is expected to implement at least `fit_transform` and `transform`
(exact API is defined by the encoders in `fedot.preprocessing.categorical_encoding`).
"""
ENCODER_MAPPING = {
    EncodingStrategyEnum.label: LabelEncoder,
    EncodingStrategyEnum.ohe: OneHotEncoder,
}
