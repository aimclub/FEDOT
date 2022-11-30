from dataclasses import dataclass
from typing import Optional

import numpy as np

from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import TaskTypesEnum


@dataclass
class SupplementaryData:
    """
    A class that stores variables for internal purposes in core during fit
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

    @property
    def compound_mask(self):
        """ The method allows to combine a mask with features in the form of
        one-dimensional array.
        """

        input_ids = np.array(self.features_mask.get('input_ids'), dtype=str)
        flow_lens = np.array(self.features_mask.get('flow_lens'), dtype=str)
        comp_list = np.core.defchararray.add(input_ids, flow_lens)
        return comp_list

    @property
    def flow_mask(self) -> list:
        return self.features_mask.get('flow_lens')

    def define_parents(self, unique_features_masks: np.array, task: TaskTypesEnum):
        """ Define which parent should be "Data parent" and "Model parent"
        for decompose operation

        :param unique_features_masks: unique values for mask
        :param task: task to solve
        """
        if not isinstance(self.previous_operations, list) or len(self.previous_operations) == 1:
            raise ValueError(f'Data was received from one node but at least two nodes are required')

        data_operations = OperationTypesRepository('data_operation').suitable_operation(task_type=task)

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
