from typing import Optional

import numpy as np

from fedot.core.operations.evaluation.operation_implementations. \
    implementation_interfaces import DataOperationImplementation
from fedot.core.repository.dataset_types import DataTypesEnum
import os
import gensim.downloader as api
from gensim.models import KeyedVectors


class PretrainedEmbeddingsImplementation(DataOperationImplementation):
    """ Class for text vectorization by pretrained embeddings
    model_name can be selected from https://github.com/RaRe-Technologies/gensim-data"""

    def __init__(self, **params: Optional[dict]):
        if not params:
            self.model_name = 'glove-twitter-25'
        else:
            self.model_name = params.get('model_name')

        self._download_model_resources()
        super().__init__()

    def fit(self, input_data):
        """ Class doesn't support fit operation

        :param input_data: data with features, target and ids to process
        """
        pass

    def transform(self, input_data, is_fit_pipeline_stage: Optional[bool]):
        """ Method for transformation of the text data

        :param input_data: data with features, target and ids to process
        :param is_fit_pipeline_stage: is this fit or predict stage for pipeline
        :return output_data: output data with transformed features table
        """

        embed_data = np.stack([self.vectorize_sum(text, self.model) for text in input_data.features])
        output_data = self._convert_to_output(input_data,
                                              embed_data,
                                              data_type=DataTypesEnum.table)
        return output_data

    @staticmethod
    def vectorize_sum(text: str, embeddings):
        """ Method converts text to a sum of token vectors

        :param text: str with text data
        :param embeddings: gensim pretrained embeddings
        :return features: one-dimensional np.array with numbers
        """
        embedding_dim = embeddings.vectors.shape[1]
        features = np.zeros([embedding_dim], dtype='float32')

        for word in text.split():
            if word in embeddings:
                features += embeddings[f'{word}']

        return features

    def _download_model_resources(self):
        """ Method for downloading text embeddings. Embeddings are loaded into external folder"""
        print('Trying to download embeddings...')
        model_path = api.load(f"{self.model_name}", return_path=True)

        if os.path.exists(model_path):
            print('Embeddings are already downloaded. Loading model...')
            self.model = KeyedVectors.load_word2vec_format(model_path, binary=False)

    def get_params(self):
        raise NotImplementedError()
