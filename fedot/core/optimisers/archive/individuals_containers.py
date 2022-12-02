# This code is modified part of DEAP library (Library URL: https://github.com/DEAP/deap).
from bisect import bisect_right
from operator import eq
from typing import Callable, Optional

from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.optimisers.opt_history_objects.individual import Individual


class HallOfFame:
    """
    The hall of fame contains the best individual that ever lived in the
    population during the evolution. It is lexicographically sorted at all
    time so that the first element of the hall of fame is the individual that
    has the best first fitness value ever seen, according to the weights
    provided to the fitness at creation time.

    The insertion is made so that old individuals have priority on new
    individuals. A single copy of each individual is kept at all time, the
    equivalence between two individuals is made by the operator passed to the
    *similar* argument.

    :param maxsize: The maximum number of individual to keep in the hall of
                    fame.
    :param similar: An equivalence operator between two individuals, optional.
                    It defaults to operator :func:`operator.eq`.

    The class :class:`HallOfFame` provides an interface similar to a list
    (without being one completely). It is possible to retrieve its length, to
    iterate on it forward and backward and to get an item or a slice from it.
    """

    def __init__(self, maxsize: Optional[int], similar: Callable = eq):
        self.maxsize = maxsize or 0
        self.keys = list()
        self.items = list()
        self.similar = similar

    def update(self, population: PopulationT):
        """
        Update the hall of fame with the *population* by replacing the
        worst individuals in it by the best individuals present in
        *population* (if they are better). The size of the hall of fame is
        kept constant.

        :param population: A list of individual with a fitness attribute to
                           update the hall of fame with.
        """
        for ind in population:
            if len(self) == 0 and self.maxsize != 0:
                # Working on an empty hall of fame is problematic for the loop
                self.insert(population[0])
                continue
            if ind.fitness > self[-1].fitness or len(self) < self.maxsize:
                for hofer in self:
                    # Loop through the hall of fame to check for any similar individual
                    if self.similar(ind, hofer):
                        break
                else:
                    # The individual is unique and strictly better than the worst
                    if len(self) >= self.maxsize:
                        self.remove(-1)
                    self.insert(ind)

    def insert(self, item: Individual):
        """
        Insert a new individual in the hall of fame using the
        :func:`~bisect.bisect_right` function. The inserted individual is
        inserted on the right side of an equal individual. Inserting a new
        individual in the hall of fame also preserve the hall of fame's order.
        This method **does not** check for the size of the hall of fame, in a
        way that inserting a new individual in a full hall of fame will not
        remove the worst individual to maintain a constant size.

        :param item: The individual with a fitness attribute to insert in the
                     hall of fame.
        """
        i = bisect_right(self.keys, item.fitness)
        self.items.insert(len(self) - i, item)
        self.keys.insert(i, item.fitness)

    def remove(self, index: int):
        """
        Remove the specified *index* from the hall of fame.

        :param index: An integer giving which item to remove.
        """
        del self.keys[len(self) - (index % len(self) + 1)]
        del self.items[index]

    def clear(self):
        """Clear the hall of fame."""
        del self.items[:]
        del self.keys[:]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def __iter__(self):
        return iter(self.items)

    def __reversed__(self):
        return reversed(self.items)

    def __str__(self):
        return str(self.items)


class ParetoFront(HallOfFame):
    """
    The Pareto front hall of fame contains all the non-dominated individuals
    that ever lived in the population. That means that the Pareto front hall of
    fame can contain an infinity of different individuals.

    :param similar: A function that tells the Pareto front whether or not two
                    individuals are similar, optional.

    The size of the front may become very large if it is used for example on
    a continuous function with a continuous domain. In order to limit the number
    of individuals, it is possible to specify a similarity function that will
    return :data:`True` if the genotype of two individuals are similar. In that
    case only one of the two individuals will be added to the hall of fame. By
    default the similarity function is :func:`operator.eq`.

    ParetoFront also supports hard-limiting maxsize, in which cases the element
    with worst primary metric is removed. By default ParetoFront is unbounded.

    Since, the Pareto front hall of fame inherits from the :class:`HallOfFame`,
    it is sorted lexicographically at every moment.
    """

    def __init__(self, maxsize: Optional[int] = None, similar: Callable = eq):
        HallOfFame.__init__(self, maxsize, similar)

    def update(self, population: PopulationT):
        """
        Update the Pareto front hall of fame with the *population* by adding
        the individuals from the population that are not dominated by the hall
        of fame. If any individual in the hall of fame is dominated it is
        removed.

        :param population: A list of individual with a fitness attribute to
                           update the hall of fame with.
        """
        for ind in population:
            is_dominated = False
            dominates_one = False
            has_twin = False
            to_remove = []
            for i, hof_member in enumerate(self):  # hall of fame member
                if not dominates_one and hof_member.fitness.dominates(ind.fitness):
                    is_dominated = True
                    break
                elif ind.fitness.dominates(hof_member.fitness):
                    dominates_one = True
                    to_remove.append(i)
                elif ind.fitness == hof_member.fitness and self.similar(ind, hof_member):
                    has_twin = True
                    break

            for i in reversed(to_remove):  # Remove the dominated hofer
                self.remove(i)
            if not is_dominated and not has_twin:
                if len(self) >= self.maxsize > 0:
                    self.remove(-1)
                self.insert(ind)
