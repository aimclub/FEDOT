import shap
from numpy import ndarray

from fedot.core.chains.chain import Chain


class ChainExplainer:
    def __init__(self, chain: Chain, data):
        self._chain = chain
        self._data = data

    def explain_with_shap(self):
        explainer = shap.KernelExplainer(self._chain_predict_wrapper, self._data.features, link="logit")
        shap_values = explainer.shap_values(self._data.features, n_sample=3)

        # visualize the first prediction's explanation
        shap.force_plot(explainer.expected_value[0], shap_values[0][0, :],
                        self._data.features.iloc[0, :], link="logit")
        return shap_values

    def _chain_predict_wrapper(self, features: ndarray):
        input_data = self._data
        input_data.features = features
        return self._chain.predict(input_data).predict
