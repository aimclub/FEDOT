from dataclasses import dataclass
from typing import Optional


@dataclass
class DataInfo:
    """
    Data class with additional info for the InputData and OutputData objects
    """
    input_ids: list = None
    flow_lens: list = None
    # Is it data in the main branch
    is_main_target: bool = True
    # Amount of nodes, which Data visited
    data_flow_length: int = 0
    # Masked features in the for data
    masked_features: Optional[list] = None

    def calculate_data_flow_len(self, outputs):
        """ Method for calculating data flow length (amount of visited nodes)

        :param outputs: list with OutputData
        """
        data_flow_lens = [output.get_flow_length for output in outputs]

        if len(data_flow_lens) == 1:
            data_flow_len = data_flow_lens[0]
        else:
            data_flow_len = max(data_flow_lens)

        # Update amount of nodes which this data have visited
        self.data_flow_length = data_flow_len + 1
