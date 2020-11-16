from datetime import timedelta

from core.composer.chain import Chain
from core.models.data import InputData
from core.models.model import Model
from core.repository.model_types_repository import ModelMetaInfo


class ChainModel(Model):
    def __init__(self, chain: Chain):
        if not chain.root_node:
            raise ValueError(f'ChainModel could not create instance of empty Chain!')

        super().__init__('chain_model')
        self.chain = chain

    def fit(self, data: InputData):
        predicted_train = self.chain.fit(input_data=data, verbose=False)
        fitted_chain_model_head = self.chain.root_node

        return fitted_chain_model_head, predicted_train.predict

    def predict(self, fitted_model, data: InputData):
        prediction = self.chain.predict(input_data=data)

        return prediction.predict

    def fine_tune(self, data: InputData, iterations: int,
                  max_lead_time: timedelta = timedelta(minutes=5)):
        self.chain.fine_tune_all_nodes(input_data=data)

    @property
    def metadata(self) -> ModelMetaInfo:
        root_node = self.chain.root_node

        model_info = ModelMetaInfo(root_node.model.metadata.id, root_node.model.metadata.input_types,
                                   root_node.model.metadata.output_types, root_node.model.metadata.task_type,
                                   ['random'], ['any'], ['atomised'])

        return model_info

    @property
    def description(self):
        model_type = self.model_type
        model_length = self.chain.length
        model_depth = self.chain.depth
        model_types = {}

        for node in self.chain.nodes:
            if node.model.model_type in model_types:
                model_types[node.model.model_type] += 1
            else:
                model_types[node.model.model_type] = 1

        return f'{model_type}_length:{model_length}_depth:{model_depth}_types{model_types}'
