from golem.utilities.data_structures import ComparableEnum as Enum
from dataclasses import dataclass
from fedot.core.backend.backend import Backend
from typing import Dict
import torch
from sentence_transformers import SentenceTransformer

from fedot.core.data.data_tools import get_idx_from_features_names


class EmbeddingMethodEnum(Enum):
    transformer = "sentence_transformer"


@dataclass
class EmbedderParameters:
    model_name: str
    batch_size: int
    device: torch.device
    method: EmbeddingMethodEnum


class TextToEmbedding:
    def __init__(self, model_name: str, device: torch.device | None = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model = SentenceTransformer(model_name, device=str(device))

    def __call__(self, sentences: list[str]) -> torch.Tensor:
        
        embeddings = self.model.encode(
            sentences,
            convert_to_numpy=False,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        return embeddings

def encode_text_features(
    X,
    parameters: EmbedderParameters,
    ) -> torch.Tensor:
    
    text_embedder = TextToEmbedding(parameters.model_name, parameters.device)

    n_samples = X.shape[0]
    num_cols = X.shape[1]
    embedding_parts = []

    for col_idx in range(num_cols):
        texts = X[:, col_idx].astype(str)
        all_embeddings = []
        for i in range(0, n_samples, parameters.batch_size):
            batch = texts[i : i + parameters.batch_size].tolist()
            embeddings = text_embedder(batch)
            all_embeddings.append(embeddings)

        embeddings_tensor = torch.cat(all_embeddings, dim=0)
        embedding_parts.append(embeddings_tensor)

    try:
        embeddings_all = torch.cat(embedding_parts, dim=1)
    except Exception as e:
        raise ValueError(f"Failed to get embeddings: {e}")
    
    embeddings_all = embeddings_all.to(torch.float32)
    return embeddings_all


def get_embedder_parameters(parameters) -> EmbedderParameters:
    if isinstance(parameters, Dict):
        if not parameters:
            parameters = EmbedderParameters(
                model_name='all-distilroberta-v1',
                batch_size=32,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                method=EmbeddingMethodEnum.transformer
            )
        
        else:
            parameters = EmbedderParameters(model_name=parameters['model_name'],
                                    batch_size=parameters['batch_size'],
                                    device=torch.device(parameters['device']),
                                    method=EmbeddingMethodEnum(parameters['method']))

        return parameters

    elif isinstance(parameters, EmbedderParameters):
        return parameters
    
    else:
        raise ValueError(f"Invalid embedderparameters type: {type(parameters)}")
    

def get_text_embeddings(features, text_idx, strategy, features_names):
    xp = Backend.xp
    device = Backend.device

    if text_idx is None:
        return None, None

    strategy = get_embedder_parameters(strategy)

    text_idx = get_idx_from_features_names(text_idx, features_names)

    text_features = features[:, text_idx]

    if strategy.method == EmbeddingMethodEnum.transformer:
        embeddings = encode_text_features(text_features, strategy)
    else:
        raise ValueError(f"Unknown embedding method: {strategy.method}")

    if strategy.device.type != device.type:
        embeddings = embeddings.to(device)

    features[:, text_idx] = xp.zeros(features[:, text_idx].shape, dtype=xp.float32)

    return embeddings, text_idx