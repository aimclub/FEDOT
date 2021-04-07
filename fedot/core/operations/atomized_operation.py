from datetime import timedelta
from uuid import uuid4

from fedot.core.data.data import InputData
from fedot.core.operations.operation import Operation
from fedot.core.repository.operation_types_repository import OperationMetaInfo, \
    atomized_operation_meta_tags, atomized_operation_type
from fedot.core.utils import make_chain_generator


class AtomizedOperation(Operation):
    """ Class which replace Operation class for AtomizedOperation object """
    def __init__(self, chain: 'Chain'):
        if not chain.root_node:
            raise ValueError(f'AtomizedOperation could not create instance of empty Chain.')

        super().__init__(operation_type=atomized_operation_type())
        self.chain = chain
        self.unique_id = uuid4()

    def fit(self, data: InputData, is_fit_chain_stage: bool = True,
            use_cache: bool = True):

        predicted_train = self.chain.fit(input_data=data)
        fitted_atomized_operation_head = self.chain.root_node

        return fitted_atomized_operation_head, predicted_train.predict

    def predict(self, fitted_operation, data: InputData,
                is_fit_chain_stage: bool, output_mode: str = 'default'):
        prediction = self.chain.predict(input_data=data, output_mode=output_mode)

        return prediction.predict

    def fine_tune(self, data: InputData, iterations: int,
                  max_lead_time: timedelta = timedelta(minutes=5)):
        # Method is not supported for now
        raise NotImplementedError()

    @property
    def metadata(self) -> OperationMetaInfo:
        generator = make_chain_generator(self.chain)
        tags = set()

        for node in generator:
            tags.update(node.operation_tags)

        root_node = self.chain.root_node
        supported_strategies = None
        allowed_positions = ['any']
        tags = list(tags)

        operation_info = OperationMetaInfo(root_node.operation.metadata.id,
                                           root_node.operation.metadata.input_types,
                                           root_node.operation.metadata.output_types,
                                           root_node.operation.metadata.task_type,
                                           supported_strategies, allowed_positions,
                                           tags)
        return operation_info

    @property
    def description(self):
        operation_type = self.operation_type
        operation_length = self.chain.length
        operation_depth = self.chain.depth
        operation_id = self.unique_id
        operation_types = {}

        for node in self.chain.nodes:
            if node.operation.operation_type in operation_types:
                operation_types[node.operation.operation_type] += 1
            else:
                operation_types[node.operation.operation_type] = 1

        return f'{operation_type}_length:{operation_length}_depth:{operation_depth}' \
               f'_types:{operation_types}_id:{operation_id}'
