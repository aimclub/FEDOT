import logging
from abc import abstractmethod

dict_of_dataset = dict
dict_of_win_list = dict


class BaseEnsemble:
    """Abstract class responsible for models ensemble

    Args:
        dataset_name: name of dataset
        feature_generator_dict: dict with feature generators

    """

    def __init__(self, dataset_name: str = None,
                 feature_generator_dict: dict = None):
        self.feature_generator_dict = feature_generator_dict
        self.dataset_name = dataset_name
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metric_dict = None
        self.prediction_proba_dict = None

    @abstractmethod
    def ensemble(self) -> dict:
        raise NotImplementedError

    def create_proba_and_metrics_dicts(
            self, modelling_results: dict) -> (dict, dict):
        """
        Method for creating dictionary with structure {'ModelName': [tensor with class probs]}
        and dictionary with structure {'ModelName':[metric values]}

        Args:
            modelling_results: dict of results from modelling

        Returns:
            tuple of dicts
        """
        prediction_proba_dict = {}
        metric_dict = {}
        for generator, generator_results in modelling_results.items():
            if not modelling_results[generator]:
                continue
            best_launch = self._select_best_launch(generator_results)
            prediction_proba = generator_results[best_launch]['class_probability']
            metrics = generator_results[best_launch]['metrics']

            prediction_proba_dict.update({generator: prediction_proba})
            metric_dict.update({generator: metrics})

        return prediction_proba_dict, metric_dict

    def _select_best_launch(self, generator_results: dict) -> dict:
        """Method for selecting best launch from modelling results

        """
        best_metric = 0
        launch = 1
        for launch_num in generator_results.keys():
            metrics = generator_results[launch_num]['metrics']
            metric_sum = metrics['roc_auc'] + metrics['f1']
            if metric_sum > best_metric:
                best_metric = metric_sum
                launch = launch_num

        return launch
