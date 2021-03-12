import cProfile
import inspect
import os
import pstats
import subprocess
import seaborn as sns
import time
import numpy as np
import matplotlib.pyplot as plt

import objgraph
from memory_profiler import memory_usage

from fedot.core.chains.chain import Chain
from fedot.core.composer.composer import Composer


class TimeProfiler:
    def __init__(self):
        self.folder = None
        self.profiler = cProfile.Profile()
        self.profiler.enable()

    def _generate_pstats(self, path: str):
        """Aggregate profiler statistics and create test.pstats."""

        self.path_stats = os.path.join(path, 'output.pstats')
        stats = pstats.Stats(self.profiler)
        stats.dump_stats(self.path_stats)

    def profile(self, path: str = None, node_percent: float = 0.5, edge_percent: float = 0.1):
        """
        Method to convert the statistics from profiler to visual representation.

        params path: path to save profiling result.
        params node_percent: eliminate nodes below this threshold [default: 0.5].
        params edge_percent: eliminate edges below this threshold [default: 0.1].

        NOTE: web-interface will open always.
        NOTE: if pass path, then graph.png will be created.
        """
        self.profiler.disable()

        self.folder = os.path.abspath(path)

        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

        self._generate_pstats(self.folder)

        # Creating the PNG call-graph with the cumulative time.
        if path is not None:
            path = os.path.join(self.folder, 'graph.png')
            subprocess.getstatusoutput(f"gprof2dot -n{node_percent} -e{edge_percent} -s -f pstats {self.path_stats} | "
                                       f"dot -Tpng -o {path} && eog {path}")

        # Opening the web interface to view the detailed profiler stats.
        subprocess.getstatusoutput(f"snakeviz {self.path_stats}")


class MemoryProfiler:
    def __init__(self, function, path: str, args=None, kwargs=None, roots=None, max_depth: int = 7):
        """
        Visual interpretation of memory usage. Create 2(two) png images.

        params function: function to profile.
        params path: path to save profiling result.
        params args []: arguments for function.
        params kwargs {}: arguments for function.
        params roots []: array with FEDOT types each of them is ROOT node in call-graph.
        params max_depth: maximum depth of graph.
        """

        if args is None:
            args = []

        if kwargs is None:
            kwargs = {}

        self.folder = os.path.abspath(path)

        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

        if roots is None:
            roots = [Chain, Composer]

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
        sns.set_palette("flare", color_codes=True)

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
