import cProfile
import os
import pstats
import subprocess

from fedot.utilities.requirements_notificator import warn_requirement

try:
    import gprof2dot
except ImportError:
    warn_requirement('gprof2dot', 'fedot[profilers]', should_raise=True)
try:
    import snakeviz
except ImportError:
    warn_requirement('snakeviz', 'fedot[profilers]', should_raise=True)


class TimeProfiler:
    """Profile code and visual interpret results of it
    """

    def __init__(self):
        self.folder = None
        self.profiler = cProfile.Profile()
        self.profiler.enable()

    def _generate_pstats(self, path: str):
        """Aggregate profiler statistics and create :obj:`output.pstats` from ``Profiler``

        Args:
            path: path to save results
        """

        self.path_stats = os.path.join(os.path.abspath(path), 'output.pstats')
        stats = pstats.Stats(self.profiler)
        stats.dump_stats(self.path_stats)

    def profile(self, path: str, node_percent: float = 0.5, edge_percent: float = 0.1, open_web: bool = False):
        """Method to convert the statistics from profiler to visual representation

        Args:
            spath: path to save profiling result
            node_percent: eliminate nodes below this threshold [default: 0.5]
            edge_percent: eliminate edges below this threshold [default: 0.1]
            open_web: boolean parametr to open web-interface
        """

        self.profiler.disable()

        self.folder = os.path.abspath(path)

        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

        self._generate_pstats(self.folder)

        # Creating the PNG call-graph with the cumulative time.
        path = os.path.join(self.folder, 'graph.png')
        subprocess.getstatusoutput(f"gprof2dot -n{node_percent} -e{edge_percent} -s -f pstats {self.path_stats} | "
                                   f"dot -Tpng -o {path}")

        # Opening the web interface to view the detailed profiler stats.
        if open_web:
            subprocess.getstatusoutput(f"snakeviz {self.path_stats}")
