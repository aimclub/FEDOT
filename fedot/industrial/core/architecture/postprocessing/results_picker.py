import logging
import os
from typing import Union

import pandas as pd

from fedot.industrial.tools.serialisation.path_lib import DS_INFO_PATH, PROJECT_PATH


class ResultsPicker:
    """Class for parsing results of experiments. It parses all experiments in ``results_of_experiments``
    folder or user-specified folder and creates dataframe table that availible for further analysis.

    Args:
        path (str): path to folder with experiments. Default is ``None`` and it means that path
                    is ``results_of_experiments``.
        launch_type (str or int): number of launch to be extracted. Default is ``max`` and it means that the best launch
                                  will be extracted.

    Examples:
        >>> from fedot.industrial.core.architecture.postprocessing.results_picker import ResultsPicker
        >>> collector = ResultsPicker(path='to_your_results_folder', launch_type='max')
        >>> metrics_df = collector.run(get_metrics_df=True)
        >>> metrics_df.to_csv('metrics.csv')

    """

    def __init__(self, path: str = None, launch_type: Union[str, int] = 'max'):
        self.exp_path = self.__get_results_path(path)
        self.launch_type = launch_type
        self.logger = logging.getLogger(self.__class__.__name__)

    def __get_results_path(self, path):
        if path:
            return path
        else:
            return os.path.join(PROJECT_PATH, 'results_of_experiments')

    def run(self, get_metrics_df: bool = False, add_info: bool = False):
        """
        Base method for parsing results of experiments.

        Returns:
            Table with results of experiments.

        """

        proba_dict, metric_dict = self.get_metrics_and_proba()

        if get_metrics_df:
            if add_info:
                metrics_df = self._create_metrics_df(metric_dict)
                datasets_info = self.get_datasets_info()
                return pd.merge(
                    metrics_df,
                    datasets_info,
                    how='left',
                    on='dataset')
            return self._create_metrics_df(metric_dict)

        return proba_dict, metric_dict

    def _create_metrics_df(self, metric_dict):
        metrics_df = pd.DataFrame()
        for ds in metric_dict.keys():
            for exp in metric_dict[ds].keys():
                metrics = metric_dict[ds][exp].to_dict(orient='records')[0]
                df = pd.DataFrame.from_dict({'dataset': ds,
                                             'experiment': exp,
                                             'f1': metrics.get('f1'),
                                             'roc_auc': metrics.get('roc_auc'),
                                             'accuracy': metrics.get('accuracy'),
                                             'precision': metrics.get('precision'),
                                             'logloss': metrics.get('logloss')}, orient='index').T
                metrics_df = pd.concat([metrics_df, df], axis=0)
        return metrics_df

    def get_metrics_and_proba(self):
        experiments = self.list_dirs(self.exp_path)
        proba_dict = {}
        metric_dict = {}
        for exp in experiments:
            exp_path = os.path.join(self.exp_path, exp)
            ds_list, metrics_list, proba_list = self.read_exp_folder(exp_path)

            if ds_list is None:
                continue

            for metric, proba, dataset in zip(
                    metrics_list, proba_list, ds_list):
                if dataset not in proba_dict.keys() and proba is not None:
                    proba_dict[dataset] = {}
                if dataset not in metric_dict.keys() and proba is not None:
                    metric_dict[dataset] = {}
                proba_dict[dataset].update({exp: proba})
                metric_dict[dataset].update({exp: metric})

        return proba_dict, metric_dict

    def read_exp_folder(self, folder):
        # datasets_path = os.path.join(self.exp_path, folder)
        datasets = self.list_dirs(folder)
        metrics_list = []
        proba_list = []
        for ds in datasets:
            ds_path = os.path.join(folder, ds)
            metrics, proba = self.read_ds_data(ds_path)
            metrics_list.append(metrics)
            proba_list.append(proba)

        return datasets, metrics_list, proba_list

    def read_ds_data(self, path):
        metrics_path = os.path.join(path, 'metrics.csv')
        proba_path = os.path.join(path, 'predicted_probs.csv')

        try:
            proba = pd.read_csv(proba_path, index_col=0)
            metrics = pd.read_csv(metrics_path, index_col=0)
        except FileNotFoundError:
            self.logger.error(f'File not found: {metrics_path}')
            return None, None
        if 'index' in metrics.columns:
            del metrics['index']
            metrics = metrics.T
            metrics = metrics.rename(columns=metrics.iloc[0])
            metrics = metrics[1:]

        return metrics, proba

    @staticmethod
    def list_dirs(path):
        """Function used instead of ``os.listdir()`` to get list of non-hidden directories.

        Args:
            path (str): Path to the directory.

        Returns:
            list: List of non-hidden directories.

        """
        path_list = []
        for f in os.listdir(path):
            # if not f.startswith('.'):
            if '.' not in f:
                path_list.append(f)
        return path_list

    @staticmethod
    def list_files(path):
        """Function used instead of ``os.listdir()`` to get list of non-hidden files.

        Args:
            path (str): Path to the directory.

        Returns:
            list: List of non-hidden files.

        """
        path_list = []
        for f in os.listdir(path):
            if os.path.isfile(path + '/' + f) and not f.startswith('.'):
                path_list.append(f)
        return path_list

    def find_best_launch(self, launch_folders):
        best_metric = 0
        launch = 1
        for _dir in self.list_dirs(launch_folders):
            if len(_dir) == 1:
                metric_path = os.path.join(launch_folders, str(
                    _dir), 'test_results', 'metrics.csv')
                metrics = pd.read_csv(metric_path, index_col=0)
                if 'index' in metrics.columns:
                    del metrics['index']
                    metrics = metrics.T
                    metrics = metrics.rename(columns=metrics.iloc[0])
                    metrics = metrics[1:]
                metric_sum = metrics['roc_auc'].values[0] + \
                    metrics['f1'].values[0]
                if metric_sum > best_metric:
                    best_metric = metric_sum
                    launch = _dir
        return launch

    def get_datasets_info(self):
        table = pd.read_json(DS_INFO_PATH)
        table = table.drop([col for col in table.columns if len(
            col) == 1] + ['Dataset_id'], axis=1)
        table.columns = list(map(str.lower, table.columns))
        table.type = table.type.str.lower()
        return table
