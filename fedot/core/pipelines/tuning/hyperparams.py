import random

from hyperopt.pyll.stochastic import sample as hp_sample

from fedot.core.log import default_log
from fedot.core.pipelines.tuning.search_space import SearchSpace


class ParametersChanger:
    """
    Class for the hyperparameters changing in the operation

    :param operation_name: name of operation to get hyperparameters for
    :param current_params: current parameters value
    """

    def __init__(self, operation_name, current_params):
        self.operation_name = operation_name
        self.current_params = current_params
        self.logger = default_log(prefix='ParametersChangerLog')

    def get_new_operation_params(self):
        """ Function return a dictionary with new parameters values """

        # Get available parameters for operation
        params_list = SearchSpace().get_operation_parameter_range(self.operation_name)

        if params_list is None:
            params_dict = None
        else:
            # Get new values for all parameters
            params_dict = self.new_params_dict(params_list)

        return params_dict

    def new_params_dict(self, params_list):
        """ Change values of hyperparameters by different ways

        :param params_list: list with hyperparameters names
        """

        _change_by_name = {'lagged': {'window_size': self._incremental_change},
                           'sparse_lagged': {'window_size': self._incremental_change}}

        params_dict = {}
        for parameter_name in params_list:
            # Get current value of the parameter
            current_value = self._get_current_parameter_value(parameter_name)

            # Perform parameter value change using appropriate function

            operation_dict = _change_by_name.get(self.operation_name)
            if operation_dict is not None:
                func = operation_dict.get(parameter_name)
            else:
                # Default changes perform with random choice
                func = self._random_change
            if func is None:
                func = self._random_change
            parameters = {'operation_name': self.operation_name,
                          'current_value': current_value}

            param_value = func(parameter_name, **parameters)
            params_dict.update(param_value)

        return params_dict

    def _get_current_parameter_value(self, parameter_name):

        if isinstance(self.current_params, str):
            # TODO 'default_params' - need to process
            current_value = None
        else:
            # Dictionary with parameters
            try:
                current_value = self.current_params.get(parameter_name)
            except Exception as exec:
                self.logger.warning(f'The following error occurred during the hyperparameter configuration.{exec}')
                current_value = None

        return current_value

    @staticmethod
    def _random_change(parameter_name, **kwargs):
        """ Randomly selects a parameter value from a specified range """

        space = SearchSpace().get_operation_parameter_range(operation_name=kwargs['operation_name'],
                                                            parameter_name=parameter_name,
                                                            label=parameter_name)
        # Randomly choose new value
        new_value = hp_sample(space)
        return {parameter_name: new_value}

    @staticmethod
    def _incremental_change(parameter_name, **kwargs):
        """ Next to the current value, the normally distributed new value is set aside """
        # TODO add the ability to limit the boundaries of the params ranges
        sigma = kwargs['current_value'] * 0.3
        new_value = random.normalvariate(kwargs['current_value'], sigma)
        return {parameter_name: new_value}
