#    This code is modified part of DEAP library (Library URL: https://github.com/DEAP/deap).
import sys
from numbers import Number

from typing import Sequence, Union
from operator import mul, truediv

import numpy as np

from fedot.core.optimisers.fitness.fitness import Fitness


class MultiObjFitness(Fitness):
    """The fitness is a measure of quality of a solution. If *values* are
    provided as a tuple, the fitness is initialized using those values,
    otherwise it is empty (or invalid).

    :param values: The initial values of the fitness as a tuple, optional.

    Fitnesses may be compared using the ``>``, ``<``, ``>=``, ``<=``, ``==``,
    ``!=``. The comparison of those operators is made lexicographically.
    Maximization and minimization are taken care off by a multiplication
    between the :attr:`weights` and the fitness :attr:`values`. The comparison
    can be made between fitnesses of different size, if the fitnesses are
    equal until the extra elements, the longer fitness will be superior to the
    shorter.

    Different types of fitnesses are created in the :ref:`creating-types`
    tutorial.

    .. note::
       When comparing fitness values that are **minimized**, ``a > b`` will
       return :data:`True` if *a* is **smaller** than *b*.
    """

    def __init__(self, values: Sequence[Number] = (), weights: Union[Sequence[Number], Number] = None):
        if weights is None:
            # Default weights
            weights = weights or (1,) * len(values)
        elif isinstance(weights, Number):
            # Single value provided
            weights = (weights,) * len(values)
        elif isinstance(weights, Sequence):
            self._check_length(values, weights)
        else:
            raise TypeError("Attribute weights of %r must be a sequence or a number." % self.__class__)

        self.weights = weights
        self.values = values

    def getValues(self):
        return tuple(map(truediv, self.wvalues, self.weights))

    def setValues(self, values):
        if values is None:
            self.reset()
        else:
            self._check_length(values)
        try:
            self.wvalues = tuple(map(mul, values, self.weights))
        except TypeError:
            _, _, traceback = sys.exc_info()
            raise TypeError("Both weights and assigned values must be a "
                            "sequence of numbers when assigning to values of "
                            "%r. Currently assigning value(s) %r of %r to a "
                            "fitness with weights %s."
                            % (self.__class__, values, type(values),
                               self.weights)).with_traceback(traceback)

    def delValues(self):
        self.wvalues = ()

    values = property(getValues, setValues, delValues,
                      ("Fitness values. Use directly ``individual.fitness.values = values`` "
                       "in order to set the fitness and ``del individual.fitness.values`` "
                       "in order to clear (invalidate) the fitness. The (unweighted) fitness "
                       "can be directly accessed via ``individual.fitness.values``."))

    def dominates(self, other: 'MultiObjFitness', selector=slice(None)):
        """Return true if each objective of *self* is not strictly worse than
        the corresponding objective of *other* and at least one objective is
        strictly better.

        :param other: Other multi-objective fitness for comparison
        :param selector: Slice indicating on which objectives the domination is
                    tested. The default value is `slice(None)`, representing
                    every objectives.
        """
        not_equal = False
        for self_wvalue, other_wvalue in zip(self.wvalues[selector], other.wvalues[selector]):
            if self_wvalue > other_wvalue:
                not_equal = True
            elif self_wvalue < other_wvalue:
                return False
        return not_equal

    @property
    def valid(self):
        """Assess if a fitness is valid or not."""
        return len(self.wvalues) != 0

    def __hash__(self):
        return hash(self.wvalues)

    def __lt__(self, other):
        return self.wvalues < other.wvalues

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.valid and other.valid and
                self.allclose(self.wvalues, other.wvalues))

    def _check_length(self, values, weights=None):
        if len(values) != len(weights or self.weights):
            raise TypeError("Attribute weights for all values must be provided.")
