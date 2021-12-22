from dataclasses import dataclass
from typing import Optional

import numpy as np

from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import TaskTypesEnum


@dataclass
class SupplementaryData:
    """
    A class that stores variables for for internal purposes in core during fit
    and predict. For instance, to manage dataflow properties.
    """
    # Is it data in the main branch
    is_main_target: bool = True
    # Amount of nodes, which Data visited
    data_flow_length: int = 0
    # Masked features for data
    features_mask: Optional[dict] = None
    # Last visited nodes
    previous_operations: Optional[list] = None
    # Is there a data was preprocessed or not
    was_preprocessed: bool = False
    # Collection with non-int indexes
    non_int_idx: Optional[list] = None
    # Dictionary with features and target column types
    column_types: Optional[dict] = None

    def calculate_data_flow_len(self, outputs):
        """ Method for calculating data flow length (amount of visited nodes)

        :param outputs: list with OutputData
        """
        data_flow_lens = [output.supplementary_data.data_flow_length for output in outputs]

        if len(data_flow_lens) == 1:
            data_flow_len = data_flow_lens[0]
        else:
            data_flow_len = max(data_flow_lens)

        # Update amount of nodes which this data have visited
        self.data_flow_length = data_flow_len + 1

    def prepare_parent_mask(self, outputs):
        """ The method for outputs from multiple parent nodes prepares a field
        with encoded values. This allow distinguishing from which ancestor the
        data was attached to. For example, a mask for two ancestors, each of
        which gives predictions in the form of a tabular data with two columns
        will look like this:
        {'input_ids': [0, 0, 1, 1],
         'flow_lens': [1, 1, 0, 0]}

        :param outputs: list with OutputData
        :return features_mask: dict with mask for features. A composite ID of
        two lists is used
            - id of parent operation order
            - amount of nodes, which data have visited
        """

        # For each parent output prepare mask
        input_ids = []
        flow_lens = []
        input_id = 0
        for output in outputs:
            predicted_values = np.array(output.predict)
            # Calculate columns
            table_shape = predicted_values.shape

            # Calculate columns
            if len(table_shape) == 1:
                features_amount = 1
            else:
                features_amount = table_shape[1]
            # Order of ancestral operations
            id_mask = [input_id] * features_amount
            input_ids.extend(id_mask)

            # Number of nodes visited by the data
            flow_mask = [output.supplementary_data.data_flow_length] * features_amount
            flow_lens.extend(flow_mask)

            # Update input id
            input_id += 1

        self.features_mask = {'input_ids': input_ids, 'flow_lens': flow_lens}

    def get_compound_mask(self):
        """ The method allows to combine a mask with features in the form of
        an one-dimensional array.
        """

        input_ids = np.array(self.features_mask.get('input_ids'), dtype=str)
        flow_lens = np.array(self.features_mask.get('flow_lens'), dtype=str)
        comp_list = np.core.defchararray.add(input_ids, flow_lens)
        return comp_list

    def get_flow_mask(self) -> list:
        return self.features_mask.get('flow_lens')

    def define_parents(self, unique_features_masks: np.array, task: TaskTypesEnum):
        """ Define which parent should be "Data parent" and "Model parent"
        for decompose operation

        :param unique_features_masks: unique values for mask
        :param task: task to solve
        """
        if not isinstance(self.previous_operations, list) or len(self.previous_operations) == 1:
            raise ValueError(f'Data was received from one node but at least two nodes are required')

        data_operations, _ = OperationTypesRepository('data_operation').suitable_operation(task_type=task)

        # Which data operations was in pipeline before decompose operation
        previous_data_operation = None
        for prev_operation in self.previous_operations:
            if prev_operation in data_operations:
                previous_data_operation = prev_operation
                # Take first data operation as "Data parent"
                break

        if previous_data_operation is not None:
            # Lagged operation by default is "Data parent"
            data_ids = np.ravel(np.argwhere(np.array(self.previous_operations) == previous_data_operation))
            data_parent_id = data_ids[0]
            model_ids = np.ravel(np.argwhere(np.array(self.previous_operations) != previous_data_operation))
            model_parent_id = model_ids[0]
        else:
            model_parent_id = 0
            data_parent_id = 1

        data_parent = unique_features_masks[data_parent_id]
        model_parent = unique_features_masks[model_parent_id]

        return model_parent, data_parent
