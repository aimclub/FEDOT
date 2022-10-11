import os
from typing import Optional

import numpy as np

from fedot.core.data.data import InputData, OutputData
from fedot.core.log import default_log
from fedot.core.operations.evaluation.operation_implementations. \
    implementation_interfaces import DataOperationImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.utilities.requirements_notificator import warn_requirement

try:
    import gensim.downloader as api
    from gensim.models import KeyedVectors
except ModuleNotFoundError:
    warn_requirement('gensim')
    api = None
    KeyedVectors = None


class PretrainedEmbeddingsImplementation(DataOperationImplementation):
    """ Class for text vectorization by pretrained embeddings
    model_name can be selected from https://github.com/RaRe-Technologies/gensim-data"""

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)
        self.logger = default_log(prefix='FEDOT logger')
        self._download_model_resources()

    def fit(self, input_data: InputData):
        """ Class doesn't support fit operation

        :param input_data: data with features, target and ids to process
        """
        pass

    def transform(self, input_data: InputData) -> OutputData:
        """ Method for transformation of the text data for predict stage

        :param input_data: data with features, target and ids to process
        :return output_data: output data with transformed features table
        """

        embed_data = np.stack([self.vectorize_avg(text, self.model) for text in input_data.features])
        output_data = self._convert_to_output(input_data,
                                              embed_data,
                                              data_type=DataTypesEnum.table)
        return output_data

    @staticmethod
    def vectorize_avg(text: str, embeddings):
        """ Method converts text to an average of token vectors

        :param text: str with text data
        :param embeddings: gensim pretrained embeddings
        :return features: one-dimensional np.array with numbers
        """
        embedding_dim = embeddings.vectors.shape[1]
        features = np.zeros([embedding_dim], dtype='float32')
        num_words = 0

        for word in text.split():
            if word in embeddings:
                features += embeddings[f'{word}']
                num_words += 1

        if num_words > 0:
            return features / num_words
        return features

    def _download_model_resources(self):
        """ Method for downloading text embeddings. Embeddings are loaded into external folder"""
        self.logger.info('Trying to download embeddings...')
        model_name = self.params.setdefault('model_name', 'glove-twitter-25')
        model_path = api.load(f"{model_name}", return_path=True)

        if os.path.exists(model_path):
            self.logger.info('Embeddings are already downloaded. Loading model...')
            self.model = KeyedVectors.load_word2vec_format(model_path, binary=False)
