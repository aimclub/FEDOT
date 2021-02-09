from typing import Optional

from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.tree import DecisionTreeRegressor


class FilterOperation:
    """ Base class for applying filtering operations on tabular data """

    def __init__(self):
        self.inner_model = None
        self.operation = None

    def fit(self, features, target):
        """ Method for fit filter

        :param features: tabular data for operation training
        :param target: target output
        :return operation: trained operation (optional output)
        """

        self.operation.fit(features, target)

        return self.operation

    def transform(self, features, is_fit_chain_stage: bool):
        """ Method for making prediction

        :param features: tabular data for filtering
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return inner_features: filtered rows
        """
        if is_fit_chain_stage:
            # For fit stage - filter data
            mask = self.operation.inlier_mask_
            inner_features = features[mask]
        else:
            # For predict stage there is a need to safe all the data
            inner_features = features

        return inner_features

    def get_params(self):
        return self.operation.get_params()


class LinearRegRANSAC(FilterOperation):
    """
    RANdom SAmple Consensus (RANSAC) algorithm with LinearRegression as core
    Task type - regression
    """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.inner_model = LinearRegression()
        self.operation = RANSACRegressor(base_estimator=self.inner_model)
        self.params = params


class NonLinearRegRANSAC(FilterOperation):
    """
    RANdom SAmple Consensus (RANSAC) algorithm with DecisionTreeRegressor as core
    Task type - regression
    """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.inner_model = DecisionTreeRegressor()
        self.operation = RANSACRegressor(base_estimator=self.inner_model)
        self.params = params
