from datetime import timedelta
from typing import Union

from core.composer.chain import Chain
from core.log import default_log, Log
from core.models.data import InputData
from utilities.synthetic.chain_template_new import ChainTemplate, ModelTemplate, extract_subtree_root


class Tune:
    def __init__(self, chain,
                 log: Log = default_log(__name__)):
        self.chain = chain
        self.chain_template = ChainTemplate(self.chain)
        self.log = log

    def fine_tune(self, input_data: InputData, iterations: int = 30,
                  max_lead_time: timedelta = timedelta(minutes=5),
                  verbose=False,
                  model_to_tune: Union[int, str] = 'primary'):
        """
        :param InputData input_data: data used to tune the model
        :param iterations: max number of iterations
        :param max_lead_time: max time available for tuning process
        :param verbose: flag used for status printing to console, default False
        :param Union[int, str] model_to_tune: node type ('primary', 'root' or 'all')
        or model number retrieved from json chain template. Default: 'primary'.
        :return Chain
        """

        if model_to_tune == 'primary':
            self._fine_tune_primary_nodes(input_data, iterations,
                                          max_lead_time, verbose)
        elif model_to_tune == 'root':
            self._fine_tune_root_node(input_data, iterations,
                                      max_lead_time, verbose)
        elif model_to_tune == 'all':
            self._fine_tune_all_nodes(input_data, iterations,
                                      max_lead_time, verbose)
        else:
            if not isinstance(model_to_tune, int):
                ex = f"Incorrect model_to_tune. Expected 'primary', 'root', 'all'" \
                     f"or int number, got {model_to_tune}"
                self.log.error(ex)
                raise ValueError(ex)
            self._fine_tune_certain_node(model_to_tune, input_data, iterations,
                                         max_lead_time, verbose)

        return self.chain

    def _fine_tune_primary_nodes(self, input_data, iterations, max_lead_time,
                                 verbose=False):
        # implementation the fine_tune_primary_nodes function
        pass

    def _fine_tune_root_node(self, input_data, iterations, max_lead_time,
                             verbose=False):
        # use current logic without recursion, something like
        # self.chain.root_node.fine_tune(recursive=False, data)
        pass

    def _fine_tune_all_nodes(self, input_data, iterations, max_lead_time,
                             verbose=False):
        # use current logic, something like
        # self.root_node.fine_tune(recursive=True, data)
        pass

    def _fine_tune_certain_node(self, model_id, input_data, iterations, max_lead_time,
                                verbose=False):
        new_root = extract_subtree_root(root_model_id=model_id,
                                        chain_template=self.chain_template)
        subchain = Chain()
        subchain.add_node(new_root)
        updated_subchain = Tune(subchain).fine_tune(model_to_tune='root',
                                                    input_data=input_data, iterations=iterations,
                                                    max_lead_time=max_lead_time, verbose=verbose)

        self._update_template(model_id=model_id,
                              updated_node=updated_subchain.root_node)

        # here chain_template converted to chain have to be but absent
        # something like
        # self.chain = self.chain_template.convert_to_chain()

    def _update_template(self, model_id, updated_node):
        model_template = [model_template for model_template in self.chain_template.model_templates
                          if model_template.model_id == model_id][0]
        update_node_template = ModelTemplate(updated_node)

        model_template.params = update_node_template.params
        model_template.fitted_model_path = update_node_template.fitted_model_path
