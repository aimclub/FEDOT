from __future__ import annotations

import inspect

from typing import TYPE_CHECKING, Any, Callable, Dict, List


if TYPE_CHECKING:
    from fedot.core.visualisation.opt_history.history_visualization import HistoryVisualization


def per_time(visualization, **kwargs):
    value = kwargs.get('per_time')
    if value and visualization.history.individuals[0][0].metadata.get('evaluation_time_iso') is None:
        visualization.log.warning('Evaluation time not found in optimization history. '
                                  'Showing fitness line per generations...')
        kwargs['per_time'] = False
    return kwargs


def best_fraction(visualization, **kwargs):
    value = kwargs.get('best_fraction')
    if value is not None and (value <= 0 or value > 1):
        raise ValueError('`value` argument should be in the interval (0, 1].')
    return kwargs


class ArgConstraintWrapper(type):
    DEFAULT_CONSTRAINTS = {
        'best_fraction': best_fraction,
        'per_time': per_time
    }

    @staticmethod
    def wrap_constraints(constraint_checkers: List[Callable[[HistoryVisualization, Dict[str, Any]], Dict[str, Any]]]):
        def decorator(visualize_function):
            """Return a wrapped instance method"""

            def outer(visualization, **kwargs):
                visualization_parameters = inspect.signature(visualize_function).parameters
                # Get default values and add them to the kwargs.
                default_kwargs = {p_name: p.default for p_name, p in visualization_parameters.items()
                                  if p.default is not p.empty}
                default_kwargs.update(kwargs)
                kwargs = default_kwargs
                # Filter wrong kwargs with warning.
                kwargs_to_ignore = []
                for argument in kwargs.keys():
                    if argument not in visualization_parameters:
                        visualization.log.warning(
                            f'Argument `{argument}` is not supported for "{visualization.__class__.__name__}". '
                            f'It is ignored.')
                        kwargs_to_ignore.append(argument)
                kwargs = {key: value for key, value in kwargs.items() if key not in kwargs_to_ignore}
                # Apply constraint_checkers iteratively.
                for checker in constraint_checkers:
                    kwargs = checker(visualization, **kwargs)
                # Make a visualization.
                visualization.log.info(
                    'Visualizing optimization history... It may take some time, depending on the history size.')
                return_value = visualize_function(visualization, **kwargs)
                return return_value

            return outer

        return decorator

    def __new__(mcs, name, bases, attrs):
        """If the class has a 'visualize' method, wrap it"""
        constraint_checkers = []
        if 'visualize' in attrs:
            parameters = inspect.signature(attrs['visualize']).parameters
            for kwarg, constraint_checker in mcs.DEFAULT_CONSTRAINTS.items():
                if kwarg in parameters:
                    constraint_checkers.append(constraint_checker)
            if 'constraint_checkers' in attrs:
                # Class-defined checkers
                for constraint_checker in attrs['constraint_checkers']:
                    constraint_checkers.append(constraint_checker)

            attrs['visualize'] = mcs.wrap_constraints(constraint_checkers)(attrs['visualize'])

        return super(ArgConstraintWrapper, mcs).__new__(mcs, name, bases, attrs)
