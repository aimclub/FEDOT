from golem.utilities.data_structures import ComparableEnum as Enum
from dataclasses import dataclass
from fedot.core.backend.backend import backend
from typing import Dict, Optional, Union, List
import torch
from sentence_transformers import SentenceTransformer

from fedot.preprocessing.preprocessor_types import EmbedderParameters
from fedot.core.data.complex_types import ArrayType


class TextToEmbedding:
    def __init__(self, model_name: str, device: Optional[torch.device] = None):
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
    X: ArrayType,
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
        raise ValueError(f"Failed to get embeddings") from e
    
    embeddings_all = embeddings_all.to(torch.float32)
    return embeddings_all
