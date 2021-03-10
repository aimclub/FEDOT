import inspect
import types
from typing import Optional

from humanfriendly import format_timespan
import networkx as nx
import pandas as pd

from fedot.core.log import default_log

pd.options.mode.chained_assignment = None


class Profiler:
    """A Profiler.

    A class for profiling the framework by time. Before starting the work with
    FEDOT, create a class. After you finish your work call the method:
    'profile(path: str)' to create a report in HTML format.

    Examples:
    --------
    >>> p = Profile()
    >>> 'some code'
    >>> p.profile('report.html')
    """

    _DROP_FUNCS = ['get_record_history_wrapper', '__repr__']

    _CSS_ELEM = '''
        <style type="text/css">

        ul, #main-ul {
          list-style-type: none;
        }

        #main-ul {
          margin: 0;
          padding: 0;
        }

        .caret {
          cursor: pointer; 
          user-select: none; 
        }

        .caret::before {
          content: "\\27A4";
          color: black;
          display: inline-block;
          margin-right: 6px;
        }

        .caret-down::before {
          content: "\\25BC";
          color: black;
          display: inline-block;
          margin-right: 6px; 
        }

        .nested {
          display: none;
        }

        .active {
          display: block;
        }
        
        .sub-items li {
          position: relative;
          margin-left: -32px;
          padding-left: 10px;
          padding-top: 2px;
          border-left: 1px dotted #0000BB;
        }
        </style>
    '''

    _JS_ELEM = '''
        <script>
        const toggler = document.getElementsByClassName("caret")
        
        for (let i = 0; i < toggler.length; i++) {
          toggler[i].addEventListener("click", function() {
            this.parentElement.querySelector(".nested").classList.toggle("active")
            this.classList.toggle("caret-down")
          })
        }
        </script>
    '''

    _GRADIENT_COLORS = ['1e9600', '2c9a00', '389e00', '43a200', '4da600',
                        '56aa00', '5fae00', '68b200', '71b600', '7aba00',
                        '82be00', '8bc200', '94c600', '9cca00', 'a5ce00',
                        'aed200', 'b7d500', 'c0d900', 'c8dd00', 'd1e000',
                        'dae400', 'e3e700', 'edeb00', 'f6ef00', 'fff200',
                        'fff200', 'ffeb00', 'ffe400', 'ffdd00', 'ffd500',
                        'ffce00', 'ffc700', 'ffbf00', 'ffb800', 'ffb000',
                        'ffa800', 'ffa100', 'ff9900', 'ff9000', 'ff8800',
                        'ff8000', 'ff7700', 'ff6e00', 'ff6400', 'ff5a00',
                        'ff4f00', 'ff4300', 'ff3500', 'ff2400', 'ff0000']

    def __init__(self, drop_funcs: Optional[list] = None):
        """Profiler init function.

        :param drop_funcs: Function names that will not be inspected.
        """

        if drop_funcs is None:
            drop_funcs = Profiler._DROP_FUNCS
        self.drop_funcs = drop_funcs + Profiler._DROP_FUNCS

        self.queue = []
        self.all_funcs = None
        self.full_stats_df = None
        self.prof_graph = None
        self.log = default_log(__name__)

        self._get_all_funcs()
        self.change_deco_settings({'enabled': True})

    def _get_all_funcs(self):
        """Get all funcs of FEDOT module to gather its statistics."""

        queue = [__import__('fedot')]

        modules = set()
        while len(queue) > 0:
            queue_elem = queue.pop(0)
            added_cnt = 0
            for el in dir(queue_elem):
                new_el = getattr(queue_elem, el)
                if isinstance(new_el, types.ModuleType) and new_el.__name__.startswith('fedot'):
                    queue.append(new_el)
                    added_cnt += 1
            if added_cnt == 0:
                modules.add(queue_elem)

        all_classes = set()
        self.all_funcs = set()

        for modul in modules:
            cls_from_module = [x[1] for x in inspect.getmembers(modul, inspect.isclass) if
                               x[1].__module__.startswith(modul.__name__)]
            all_classes.update(cls_from_module)
            funcs_from_module = [x[1] for x in inspect.getmembers(modul, inspect.isfunction) if
                                 x[1].__module__.startswith(modul.__name__)]
            self.all_funcs.update(funcs_from_module)
            meth_from_module = [x[1] for x in inspect.getmembers(modul, inspect.ismethod) if
                                x[0] not in self.drop_funcs and x[1].__module__.startswith(modul.__name__)]
            self.all_funcs.update(meth_from_module)

        for cls in all_classes:
            cls_name = cls.__name__.split('.')[-1]
            funcs_from_class = [x[1] for x in inspect.getmembers(cls, inspect.isfunction) if
                                x[1].__qualname__.startswith(cls_name)]
            self.all_funcs.update(funcs_from_class)
            meth_from_module = [x[1] for x in inspect.getmembers(cls, inspect.ismethod) if
                                x[0] not in self.drop_funcs and x[1].__qualname__.startswith(cls_name)]
            self.all_funcs.update(meth_from_module)

        self.all_funcs = sorted(list(self.all_funcs), key=lambda x: x.__module__ + '.' + x.__qualname__)
        self.all_funcs = [f for f in self.all_funcs if f.__name__ not in self.drop_funcs]

        self.log.info('ALL_FUNCS len = {}'.format(len(self.all_funcs)))

    def _aggregate_stats_from_functions(self):
        """Gather stats from all found functions into one dataframe."""

        cols_df = ['call_num', 'elapsed_secs', 'timestamp',
                   'prefixed_func_name', 'caller_chain']
        dfs_arr = []
        for f in self.all_funcs:
            if not hasattr(f, 'stats'):
                self.log.info('\t Func with no stats - {}'.format(f))
                continue
            curr_df = pd.DataFrame([[getattr(el, col) for col in cols_df]
                                    for el in f.stats.history], columns=cols_df)

            if len(curr_df) > 0:
                if curr_df['call_num'].value_counts().values[0] > 1:
                    curr_df = curr_df.sort_values('timestamp', kind='mergesort').reset_index(drop=True)
                    curr_df['call_num'] = list(range(1, len(curr_df) + 1))
                curr_df.insert(4, 'run_fname',
                               curr_df['prefixed_func_name'].astype(str) + ' [' + curr_df['call_num'].astype(str) + ']')
                curr_df['caller_chain'] = curr_df['caller_chain'].map(lambda x: x[-1])
                dfs_arr.append(curr_df)
            f.stats.clear_history()

        if len(dfs_arr) == 0:
            self.log.info('There is no info from functions to profile... Abort')
            return

        self.full_stats_df = pd.concat(dfs_arr)
        self.full_stats_df = self.full_stats_df.sort_values(['timestamp', 'call_num']).reset_index(drop=True)

        self.log.info('FULL_STATS_DF shape = {}'.format(self.full_stats_df.shape))

    def _generate_and_check_calls_graph(self):
        """Build graphs from functions calls and gather together."""

        self.prof_graph = nx.Graph()
        self.prof_graph.add_edges_from(list(zip(self.full_stats_df['caller_chain'].values,
                                                self.full_stats_df['run_fname'].values)))

        cc = list(nx.connected_components(self.prof_graph))
        self.full_stats_df['level'] = None

        for i in range(len(cc)):
            curr_df = self.full_stats_df.loc[self.full_stats_df.caller_chain.isin(cc[i])]
            curr_g = nx.subgraph(self.prof_graph, cc[i])
            path_lens = {x: len(y) - 1
                         for x, y in nx.shortest_path(curr_g, source=curr_df.caller_chain.values[0]).items()}
            curr_df.loc[:, 'level'] = curr_df['run_fname'].map(path_lens)
            self.full_stats_df.loc[list(curr_df.index), 'level'] = curr_df['level'].values

        self.full_stats_df = self.full_stats_df.sort_values(['timestamp', 'call_num', 'level'],
                                                            kind='mergesort').reset_index(drop=True)

    def _create_html_report(self, report_path: str, brief=False):
        """Create HTML report for FEDOT profiling."""

        df = self.full_stats_df[['run_fname', 'level', 'elapsed_secs']]

        if brief:
            total_time = sum(df[df['level'] == 1]['elapsed_secs'])
            df = df[df['elapsed_secs'] / total_time >= 0.01]

        df = pd.concat([pd.DataFrame({'run_fname': ['FEDOT ROOT'], 'level': [0], 'elapsed_secs': [0.0]}), df])
        df = df.reset_index(drop=True)

        nodes = list(df['run_fname'].values)
        levels = df['level'].values
        times = df['elapsed_secs'].values

        times[0] = sum(times[levels == 1])
        nodes[0] = f'<span><b>{format_timespan(times[0])}</b>, {nodes[0]} </span>'

        for i in range(1, len(times)):
            span = '''
            <span style="color:#000000; background: #{hex_number};";>
                <b>{sec:.2f}%</b>
            </span>
            <span style="color:#000000;";>{text}</span>
            '''
            times[i] = times[i] * 100 / times[0]
            hex_number = Profiler._GRADIENT_COLORS[int(times[i] // 2 - 0.001)]
            nodes[i] = span.format(hex_number=hex_number, sec=times[i], text=nodes[i])

        with open(report_path, 'w') as f_out:
            f_out.write(Profiler._CSS_ELEM + '\n')
            f_out.write('<body>\n')
            f_out.write('<ul id="main-ul">' + '\n')

            for i in range(len(nodes) - 1):
                if levels[i] < levels[i + 1]:
                    li = '<li><span class="caret">{}</span><ul class="nested sub-items">'
                    f_out.write('\t' * int(levels[i]) + li.format(nodes[i]) + '\n')
                elif levels[i] == levels[i + 1]:
                    li = '<li>&emsp;&nbsp;{}</li>'
                    f_out.write('\t' * int(levels[i]) + li.format(nodes[i]) + '\n')
                else:
                    li = '<li>&emsp;&nbsp;{}</li>'
                    f_out.write('\t' * int(levels[i]) + li.format(nodes[i]) + '\n')

                    for t in range(int(levels[i]), int(levels[i + 1]), -1):
                        f_out.write('\t' * (t - 1) + '</ul></li>' + '\n')

            for t in range(levels[-1], 0, -1):
                f_out.write('\t' * (t - 1) + '</ul></li>' + '\n')

            f_out.write('</ul>')
            f_out.write(Profiler._JS_ELEM + '\n')
            f_out.write('</body>\n')

    def profile(self, report_path: str = './profile_report.html'):
        """Create profile of algorithm.

        :params report_path: path to save profile.
        """

        self._aggregate_stats_from_functions()

        if self.full_stats_df is not None:
            self._generate_and_check_calls_graph()
            self._create_html_report(report_path)

    def change_deco_settings(self, new_settings: dict):
        """Update profiling deco settings.

        :params new_settings: dict with new key-values for decorator.
        """

        for f in self.all_funcs:
            if hasattr(f, 'record_history_settings'):
                for k in new_settings:
                    f.record_history_settings[k] = new_settings[k]
