from typing import Optional, Sequence
import torch
from sentence_transformers import SentenceTransformer

from fedot.core.backend.backend import Backend, torch_to_xp
from fedot.core.data.prepared_data import PreparedData
from fedot.preprocessing.methods.abstract import AbstractPreprocessingHandler


"""
How to add a new embedding method
-----------------------------------
1. Implement a new embedding function in this module (or another module) with a
   compatible signature, e.g.:
   `def my_embedding_fn(X: ArrayType, parameters: EmbedderParameters) -> torch.Tensor: ...`
2. Add a new value to :class:`EmbeddingMethodEnum` in
   `fedot/preprocessing/preprocessor_types.py` (e.g. `my_method = "my_method"`).
3. Update `fedot/core/repository/preprocessor_mapping.py` by adding a new entry to
   `EMBEDDING_METHOD_MAPPING`, e.g.:
   `EmbeddingMethodEnum.my_method: my_embedding_fn`.
"""


class TextToEmbedding:
    """
    A callable wrapper converting a list of sentences/text strings into embeddings.

    Internally it uses `SentenceTransformer` and returns embeddings as a
    `torch.Tensor`.
    """
    def __init__(self, model_name: str, device: Optional[torch.device] = None):
        """
        Args:
            model_name (str): Name or path of the SentenceTransformer model.
            device (Optional[torch.device]): Device to run the model on. If `None`,
                CUDA is used when available, otherwise CPU is used.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model = SentenceTransformer(model_name, device=str(device))

    def __call__(self, sentences: list[str]) -> torch.Tensor:
        """
        Compute embeddings for the provided sentences.

        Args:
            sentences (list[str]): Text inputs.

        Returns:
            torch.Tensor: Embeddings tensor returned by `SentenceTransformer`.
        """
        
        embeddings = self.model.encode(
            sentences,
            convert_to_numpy=False,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        return embeddings


class TransformerEmbedder(AbstractPreprocessingHandler):

    def __init__(self, 
                 model_name: str, 
                 batch_size: int = 4, 
                 device: Optional[torch.device] = Backend().device):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

        self.features_idx: Optional[Sequence[int]] = None

    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        self.features_idx = features_idx
        return self

    def transform(self, data: PreparedData) -> PreparedData:
        if self.features_idx is None:
            raise ValueError("TransformerEmbedder must be fitted first")

        xp = Backend().xp
        device = Backend().device

        features = data.features

        X = features[:, self.features_idx]

        text_embedder = TextToEmbedding(self.model_name, 
                                        self.device)

        n_samples = X.shape[0]
        num_cols = X.shape[1]
        embedding_parts = []

        for col_idx in range(num_cols):
            texts = X[:, col_idx].astype(str)
            all_embeddings = []
            for i in range(0, n_samples, self.batch_size):
                batch = texts[i : i + self.batch_size].tolist()
                embeddings = text_embedder(batch)
                all_embeddings.append(embeddings)

            embeddings_tensor = torch.cat(all_embeddings, dim=0)
            embedding_parts.append(embeddings_tensor)

        try:
            embeddings_all = torch.cat(embedding_parts, dim=1)
        except Exception as e:
            raise ValueError(f"Failed to get embeddings") from e
        
        embeddings_all = embeddings_all.to(torch.float32)

        if self.device.type != device.type:
            embeddings_all = embeddings_all.to(device)
        
        embeddings_all = torch_to_xp(embeddings_all, xp)
        
        features = xp.delete(features, self.features_idx, axis=1)
        features = xp.hstack((features, embeddings_all))

        data.features = features

        return data
