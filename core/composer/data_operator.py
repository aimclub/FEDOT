from core.models.data import InputData


class DataOperator:
    def __init__(self, external_source, type):
        self.external = external_source
        self.type = type
        self._out = None

        self.attached_node = None

    def input(self):
        if self.type == 'external':
            return self.external
        elif self.external is None and self.type == 'from_parents':
            parent_nodes = self.attached_node._nodes_from_with_fixed_order()

            parent_results = []
            for parent in parent_nodes:
                prediction = parent.operator.get_output()
                parent_results.append(prediction)
            target = parent_nodes[0].operator.external.target

            secondary_input = InputData.from_predictions(outputs=parent_results,
                                                         target=target)
            return secondary_input

    def set_output(self, output):
        self._out = output

    def get_output(self):
        if self._out is None:
            raise ValueError('Refresh data from the attached node')


def _combine_parents_simple(parent_nodes,
                            input_data,
                            parent_operation: str):
    target = input_data.target
    parent_results = []
    for parent in parent_nodes:
        if parent_operation == 'predict':
            prediction = parent.predict(input_data=input_data)
            parent_results.append(prediction)
        elif parent_operation == 'fit':
            prediction = parent.fit(input_data=input_data)
            parent_results.append(prediction)
        else:
            raise NotImplementedError()

    return parent_results, target
