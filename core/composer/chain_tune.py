from datetime import timedelta

from core.composer.chain import Chain
from core.composer.node import PrimaryNode
from core.log import default_log, Log
from core.models.data import InputData
from utilities.synthetic.chain_template_new import ChainTemplate, \
    ModelTemplate, extract_subtree_root
from functools import wraps


def tune_log_decorator(type_ex):
    def decorator(method):
        @wraps(method)
        def wrapper(ref, *args, **kwargs):
            ref.log.info(f'Start tuning {type_ex} nodes')
            value = method(ref, *args, **kwargs)
            ref.log.info(f'End tuning {type_ex} nodes')
            return value

        return wrapper

    return decorator


def log_decorator(dec, dec_param):
    def wrapper(method):
        def inner_wrapper(ref, *args, **kwargs):
            if ref.verbose:
                value = dec(dec_param)(method)(ref, *args, **kwargs)
            else:
                value = method(ref, *args, **kwargs)

            return value

        return inner_wrapper

    return wrapper


class Tune:

    def __init__(self, chain,
                 log: Log = default_log(__name__), verbose=False):
        self.chain = chain
        self.chain_template = ChainTemplate(self.chain)
        self.log = log
        self.verbose = verbose

    @log_decorator(tune_log_decorator, 'primary')
    def fine_tune_primary_nodes(self, input_data: InputData, iterations: int = 30,
                                max_lead_time: timedelta = timedelta(minutes=5),
                                verbose=False):

        """
        Optimize hyperparameters of models in primary nodes

        :param input_data: data used for tuning
        :param iterations: max number of iterations
        :param max_lead_time: max time available for tuning process
        :param verbose: flag used for status printing to console, default False
        :return: updated chain object
        """

        # if verbose:
        #     self.log.info('Start tuning of primary nodes')

        all_primary_nodes = [node for node in self.chain.nodes if isinstance(node, PrimaryNode)]
        for node in all_primary_nodes:
            node.fine_tune(input_data, max_lead_time=max_lead_time, iterations=iterations)

        # if verbose:
        #     self.log.info('End tuning')

        return self.chain

    def fine_tune_root_node(self, input_data: InputData, iterations: int = 30,
                            max_lead_time: timedelta = timedelta(minutes=5),
                            verbose=False):
        """
        Optimize hyperparameters in the root node

        :param input_data: data used for tuning
        :param iterations: max number of iterations
        :param max_lead_time: max time available for tuning process
        :param verbose: flag used for status printing to console, default False
        :return: updated chain object
        """
        if verbose:
            self.log.info('Start tuning of chain')

        node = self.chain.root_node
        node.fine_tune(input_data=input_data, max_lead_time=max_lead_time,
                       iterations=iterations, recursive=False)

        if verbose:
            self.log.info('End tuning')

        return self.chain

    def fine_tune_all_nodes(self, input_data: InputData, iterations: int = 30,
                            max_lead_time: timedelta = timedelta(minutes=5),
                            verbose=False):
        """
        Optimize hyperparameters of models in all nodes

        :param input_data: data used for tuning
        :param iterations: max number of iterations
        :param max_lead_time: max time available for tuning process
        :param verbose: flag used for status printing to console, default False
        :return: updated chain object
        """
        if verbose:
            self.log.info('Start tuning of chain')

        node = self.chain.root_node
        node.fine_tune(input_data, max_lead_time=max_lead_time, iterations=iterations, recursive=True)

        if verbose:
            self.log.info('End tuning')

        return self.chain

    def fine_tune_certain_node(self, model_id, input_data: InputData, iterations: int = 30,
                               max_lead_time: timedelta = timedelta(minutes=5),
                               verbose=False):
        """
        Optimize hyperparameters of models in the certain node,
        defined by model id

        :param int model_id: number of the certain model in the chain.
        Look for it in exported json file of your model.
        :param input_data: data used for tuning
        :param iterations: max number of iterations
        :param max_lead_time: max time available for tuning process
        :param verbose: flag used for status printing to console, default False
        :return: updated chain object
        """

        subchain = Chain()
        new_root = extract_subtree_root(root_model_id=model_id,
                                        chain_template=self.chain_template)
        subchain.add_node(new_root)
        subchain.fit(input_data=input_data, use_cache=False)

        updated_subchain = Tune(subchain).fine_tune_root_node(input_data=input_data, iterations=iterations,
                                                              max_lead_time=max_lead_time, verbose=verbose)

        self._update_template(model_id=model_id,
                              updated_node=updated_subchain.root_node)

        updated_chain = Chain()
        self.chain_template.convert_to_chain(chain_to_convert_to=updated_chain)

        return updated_chain

    def _update_template(self, model_id, updated_node):
        model_template = [model_template for model_template in self.chain_template.model_templates
                          if model_template.model_id == model_id][0]
        update_node_template = ModelTemplate(updated_node, chain_id=self.chain_template.unique_chain_id)

        model_template.params = update_node_template.params
        model_template.fitted_model_path = update_node_template.fitted_model_path
