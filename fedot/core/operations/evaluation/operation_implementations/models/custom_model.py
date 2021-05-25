import timeit
from typing import Optional
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import ModelImplementation

class CustomModelImplementation(ModelImplementation):
    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.params = params
        self.chain = None

    def fit(self, input_data):
        node_lagged_1 = PrimaryNode('lagged', node_data={'fit': input_data[0][0],
                                                         'predict': input_data[2][0]})

        node_exog = PrimaryNode('exog', node_data={'fit': input_data[1][0],
                                                   'predict': input_data[3][0]})

        node_final = SecondaryNode('ridge', nodes_from=[node_lagged_1, node_exog])
        self.chain = Chain(node_final)

        start_time = timeit.default_timer()
        self.chain.fit_from_scratch()
        amount_of_seconds = timeit.default_timer() - start_time

        print(f'\nIt takes {amount_of_seconds:.2f} seconds to train chain\n')
        #self.chain.show()

    def get_params(self):
        return self.params

    def predict(self):
        predicted_values = self.chain.predict()
        predicted_values = predicted_values.predict
        return predicted_values


