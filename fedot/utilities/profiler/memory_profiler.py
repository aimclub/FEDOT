import inspect
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from fedot.core.composer.composer import Composer
from fedot.core.pipelines.pipeline import Pipeline
from fedot.utilities.requirements_notificator import warn_requirement

try:
    import objgraph

    from memory_profiler import memory_usage
except ImportError:
    warn_requirement('objgraph', 'fedot[profilers]', should_raise=True)


class MemoryProfiler:
    """Visual interpretation of memory usage. Create two ``png`` files.

    Args:
        function: function to profile.
        path: path to save profiling result.
        list args: arguments for function in array format.
        dict kwargs: arguments for function in dictionary format.
        list roots: array with FEDOT types each of them is ROOT node in call-graph.
        int max_depth: maximum depth of graph.
    """

    def __init__(self, function, path: str, args=None, kwargs=None, roots=None, max_depth: int = 7,
                 visualization=False):
        if args is None:
            args = []

        if kwargs is None:
            kwargs = {}

        self.folder = os.path.abspath(path)

        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

        if roots is None:
            roots = [Pipeline, Composer]

        if visualization:
            # Create the plot of the memory time dependence.
            self._create_memory_plot(function, args, kwargs)

            # Create call graph.
            self._create_memory_graph(roots, max_depth)

    def _create_memory_plot(self, function, args, kwargs, interval: float = 0.1):
        start_time = time.time()
        mem_res = memory_usage((function, args, kwargs), interval=interval)
        total_time = time.time() - start_time

        length = len(mem_res)
        division = total_time / length
        time_split = np.linspace(division, total_time, length)

        max_index = np.argmax(mem_res)

        plt.figure(figsize=(12, 8))

        sns.set_style("whitegrid")

        ax = sns.lineplot(time_split, mem_res, linewidth=3, marker="X", markersize=8)
        ax.axhline(mem_res[max_index], ls='--', color='red')
        ax.axvline(time_split[max_index], ls='--', color='red')

        plt.legend(labels=[f'interval {interval}', 'max memory usage'])

        ax.set_title('Memory time dependence', size=25)

        ax.set_xlabel("time [s]", fontsize=16)
        ax.set_ylabel("memory used [KB]", fontsize=16)

        filename = os.path.join(self.folder, 'memory_plot.png')
        ax.figure.savefig(filename)

    def _create_memory_graph(self, roots, max_depth):
        filename = os.path.join(self.folder, 'memory_graph.png')

        objgraph.show_refs(
            roots,
            max_depth=max_depth,
            filename=filename,
            filter=lambda x: not type(x).__name__ in ['module', 'str', 'code', 'IPythonKernel', '_Helper',
                                                      'builtin_function_or_method', 'type', '_Printer', 'Printer',
                                                      'float64', 'float', 'int'],
            highlight=lambda x: inspect.isclass(x) or inspect.isfunction(x),
            refcounts=True
        )
