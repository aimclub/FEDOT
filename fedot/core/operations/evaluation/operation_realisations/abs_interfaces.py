from abc import abstractmethod, ABC
from typing import Optional
import numpy as np


class OperationRealisation(ABC):
    """ Interface for operations realisations methods
    Contains abstract methods, which should be implemented for applying EA
    optimizer on it
    """

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, features):
        """ Method fit operation on a dataset

        :param features: features to process
        """
        raise NotImplementedError()

    @abstractmethod
    def transform(self, features, is_fit_chain_stage: Optional[bool]):
        """ Method apply transform operation on a dataset

        :param features: features to process
        :param is_fit_chain_stage: is this fit or predict stage for chain
        """
        raise NotImplementedError()

    @abstractmethod
    def get_params(self):
        """ Method return parameters, which can be optimized for particular
        operation
        """
        raise NotImplementedError()


class EncodedInvariantOperation(OperationRealisation):
    """ Class for processing data without transforming encoded features.
    Encoded features - features after OneHot encoding operation
    """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.operation = None
        self.params = params

    def fit(self, features):
        """ Method for fit transformer with automatic determination
        of boolean features, with which there is no need to make transformation

        :param features: tabular data for operation training
        :return encoder: trained transformer (optional output)
        """

        bool_ids, ids_to_process = reasonability_check(features)
        self.ids_to_process = ids_to_process
        self.bool_ids = bool_ids

        if len(ids_to_process) > 0:
            features_to_process = np.array(features[:, ids_to_process])
            self.operation.fit(features_to_process)
        else:
            pass

        return self.operation

    def transform(self, features, is_fit_chain_stage: Optional[bool]):
        """
        The method that transforms the source features using "operation"

        :param features: tabular data for transformation
        :return transformed_features: transformed features table
        :param is_fit_chain_stage: is this fit or predict stage for chain
        """

        if len(self.ids_to_process) > 0:
            transformed_features = self._make_new_table(features)
        else:
            transformed_features = features

        return transformed_features

    def _make_new_table(self, features):
        """
        The method creates a table based on transformed data and source boolean
        features

        :param features: tabular data for processing
        :return transformed_features: transformed features table
        """

        features_to_process = np.array(features[:, self.ids_to_process])
        transformed_part = self.operation.transform(features_to_process)

        # If there are no binary features in the dataset
        if len(self.bool_ids) == 0:
            transformed_features = transformed_part
        else:
            # Stack transformed features and bool features
            bool_features = np.array(features[:, self.bool_ids])
            frames = (bool_features, transformed_part)
            transformed_features = np.hstack(frames)

        return transformed_features

    def get_params(self):
        return self.operation.get_params()


def reasonability_check(features):
    """
    Method for checking which columns contain boolean data

    :param features: tabular data for check
    :return bool_ids: indices of boolean columns in table
    :return non_bool_ids: indices of non boolean columns in table
    """
    # TODO perhaps there is a more effective way to do this
    source_shape = features.shape
    columns_amount = source_shape[1]

    bool_ids = []
    non_bool_ids = []
    # For every column in table make check
    for column_id in range(0, columns_amount):
        column = features[:, column_id]
        if len(np.unique(column)) > 2:
            non_bool_ids.append(column_id)
        else:
            bool_ids.append(column_id)

    return bool_ids, non_bool_ids
