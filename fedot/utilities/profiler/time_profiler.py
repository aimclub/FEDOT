import os
import cProfile
import pstats
import subprocess


try:
    import gprof2dot
    import snakeviz
except ImportError:
    raise ImportError("Required packages is not installed on your system."
                      " It is required to run this example."
                      " Install 'fedot/utilities/profiler/requirements_time_profiler.txt'")


class TimeProfiler:
    """
    Profile code and visual interpret results of it.
    """
    def __init__(self):
        self.folder = None
        self.profiler = cProfile.Profile()
        self.profiler.enable()

    def _generate_pstats(self, path: str):
        """
        Aggregate profiler statistics and create 'output.pstats' from cPorifler.

        :param str path: path to save results.
        """

        self.path_stats = os.path.join(os.path.abspath(path), 'output.pstats')
        stats = pstats.Stats(self.profiler)
        stats.dump_stats(self.path_stats)

    def profile(self, path: str, node_percent: float = 0.5, edge_percent: float = 0.1, open_web: bool = False):
        """
        Method to convert the statistics from profiler to visual representation.

        :param str path: path to save profiling result.
        :param float node_percent: eliminate nodes below this threshold [default: 0.5].
        :param float edge_percent: eliminate edges below this threshold [default: 0.1].
        :param bool open_web: boolean parametr to open web-interface.
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
