import numpy as np
from dataclasses import dataclass
from typing import Optional


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
    # Masked features in the for data
    features_mask: Optional[dict] = None

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
