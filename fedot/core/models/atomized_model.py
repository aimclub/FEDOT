from datetime import timedelta
from uuid import uuid4

from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData
from fedot.core.models.model import Model
from fedot.core.repository.model_types_repository import ModelMetaInfo


class AtomizedModel(Model):
    def __init__(self, chain: 'Chain'):
        if not chain.root_node:
            raise ValueError(f'AtomizedModel could not create instance of empty Chain!')

        super().__init__(model_type='atomized_model')
        self.chain = chain
        self.unique_id = uuid4()

    def fit(self, data: InputData):
        predicted_train = self.chain.fit(input_data=data, verbose=False)
        fitted_atomized_model_head = self.chain.root_node

        return fitted_atomized_model_head, predicted_train.predict

    def predict(self, fitted_model, data: InputData,  output_mode: str = 'default'):
        prediction = self.chain.predict(input_data=data, output_mode=output_mode)

        return prediction.predict

    def fine_tune(self, data: InputData, iterations: int,
                  max_lead_time: timedelta = timedelta(minutes=5)):
        from fedot.core.chains.chain_tune import Tune
        self.chain = Tune(self.chain, verbose=True).fine_tune_root_node(input_data=data,
                                                                        max_lead_time=max_lead_time,
                                                                        iterations=iterations)

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
        model_id = self.unique_id
        model_types = {}

        for node in self.chain.nodes:
            if node.model.model_type in model_types:
                model_types[node.model.model_type] += 1
            else:
                model_types[node.model.model_type] = 1

        return f'{model_type}_length:{model_length}_depth:{model_depth}_types:{model_types}_id:{model_id}'
