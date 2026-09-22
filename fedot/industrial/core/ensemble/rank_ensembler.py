import logging

from fedot.industrial.core.architecture.settings.computational import backend_methods as np
import pandas as pd

from fedot.industrial.core.metrics.evaluation import PerformanceAnalyzer
from fedot.industrial.tools.loader import DataLoader
from fedot.industrial.core.ensemble.base_ensembler import BaseEnsemble


class RankEnsemble(BaseEnsemble):
    """Class responsible for the results of ensemble models by ranking them and
    recursively adding them to the final composite model.

    Args:
        dataset_name: name of dataset
        proba_dict: dictionary with prediction probabilities
        metric_dict: dictionary with metrics

    """

    def __init__(self, dataset_name: str, proba_dict, metric_dict):
        super().__init__(dataset_name=dataset_name)
        self.best_base_results = None
        self.proba_dict = proba_dict
        self.metric_dict = metric_dict

        self.performance_analyzer = PerformanceAnalyzer()
        self.experiment_results = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.best_ensemble_metric = 0

        self.ensemble_strategy_dict = {'MeanEnsemble': np.mean,
                                       'MedianEnsemble': np.median,
                                       'MinEnsemble': np.min,
                                       'MaxEnsemble': np.max,
                                       'ProductEnsemble': np.prod}

        self.ensemble_strategy = self.ensemble_strategy_dict.keys()
        self.strategy_exclude_list = ['WeightedEnsemble']

    def ensemble(self) -> dict:
        """Returns dictionary with ranking ensemble results

        The process of ensemble consists of 3 stages. At the first stage, a dictionary is created
        that contains the name of the model as a key and the best metric value for this dataset as a value.
        The second stage is the creation of a ranked list in the form of a dictionary (self.sorted_dict),
        also at this stage parameters such as the best model and the best value of the quality metric are determined,
        which are stored in the dictionary self.best_base_results. The third stage is iterative, in accordance
        with the assigned rank, adding models to a single composite model and ensemble their predictions.

        Returns:
            Fitted Fedot pipeline with baseline model

        """
        (_, _), (_, self.test_target) = DataLoader(
            self.dataset_name).load_data()

        model_rank_dict = self._create_models_rank_dict(
            self.proba_dict, self.metric_dict)
        self.best_base_results = self._sort_models(model_rank_dict)

        return self.__iterative_model_selection(self.proba_dict)

    def _deep_search_in_dict(self, obj, key):
        if key in obj:
            return obj[key]
        for k, v in obj.items():
            if isinstance(v, dict):
                item = self._deep_search_in_dict(v, key)
                if item is not None:
                    return item

    def _create_models_rank_dict(
            self,
            prediction_proba_dict,
            metric_dict) -> dict:
        """
        Method that returns a dictionary with the best metric values of base models

        Returns:
            dictionary with structure {'ModelName': best metric values}

        """
        model_rank = {}
        for model in metric_dict[self.dataset_name].keys():
            self.logger.info(
                f'BASE RESULT FOR MODEL - {model}'.center(50, '-'))
            if prediction_proba_dict[self.dataset_name][model].shape[1] == 1:
                self.metric = 'roc_auc'
                _type = 'binary'
            else:
                self.metric = 'f1'
                _type = 'multiclass'
            self.logger.info(
                f'TYPE OF ML TASK - {_type}. Metric - {self.metric}'.center(50, '-'))
            current_metrics = metric_dict[self.dataset_name][model]
            if isinstance(current_metrics, pd.DataFrame):
                current_metrics = current_metrics.loc[0].to_dict()
            self.logger.info(current_metrics)
            model_rank.update({model: current_metrics[self.metric]})
        return model_rank

    def _sort_models(self, model_rank) -> dict:
        """
        Method that returns sorted dictionary with models results ``

        Args:
            model_rank: dictionary with structure {'ModelName': 'best metric values'}

        Returns:
            sorted dictionary with structure {'Base_model': 'best metric values'}

        """

        self.sorted_dict = dict(
            sorted(model_rank.items(), key=lambda x: x[1], reverse=True))
        self.n_models = len(self.sorted_dict)

        best_base_model = list(self.sorted_dict)[0]
        best_metric = self.sorted_dict[best_base_model]

        self.logger.info(
            f'CURRENT BEST METRIC - {best_metric}. MODEL - {best_base_model}'.center(50, '-'))
        self.experiment_results.update(
            {'Base_model': best_base_model, 'Base_metric': best_metric})
        return {'Base_model': best_base_model, 'Base_metric': best_metric}

    def __iterative_model_selection(self, prediction_proba_dict):
        """
        Method that iteratively adds models to a single composite model and ensemble their predictions

        Returns:
            dictionary with structure {'Ensemble_models': 'best ensemble metric',
                                        'Base_model': 'best base model metric'}

        """
        for top_K_models in range(1, self.n_models):

            modelling_results_top = {k: v for k, v in prediction_proba_dict[self.dataset_name].items(
            ) if k in list(self.sorted_dict.keys())[:top_K_models + 1]}
            self.logger.info(
                f'Applying ensemble {self.ensemble_strategy} strategy for {top_K_models + 1} models')

            ensemble_results = self.agg_ensemble(
                modelling_results=modelling_results_top, single_mode=True)

            top_ensemble_dict = self.__select_best_ensemble_method(
                ensemble_results)

            if len(top_ensemble_dict) == 0:
                self.logger.info(
                    f'No improvement accomplished for {list(modelling_results_top.keys())} combination')
            else:
                current_ensemble_method = list(top_ensemble_dict)[0]
                best_ensemble_metric = top_ensemble_dict[current_ensemble_method]
                model_combination = list(modelling_results_top)[
                    :top_K_models + 1]
                self.logger.info(
                    f'Accomplished metric improvement by {current_ensemble_method}:'
                    f'New best metric – {best_ensemble_metric}')

                if self.best_ensemble_metric > 0:
                    self.experiment_results.update(
                        {'Ensemble_models': model_combination})
                    self.experiment_results.update(
                        {'Ensemble_method': current_ensemble_method})
                    self.experiment_results.update(
                        {'Best_ensemble_metric': best_ensemble_metric})
        return self.experiment_results

    def __select_best_ensemble_method(self, ensemble_results: dict):
        """
        A method that iteratively searches for an ensemble algorithm that improves the current best result

        Returns:
            sorted dictionary with structure {'Ensemble_models': 'best ensemble metric'}

        """
        top_ensemble_dict = {}
        for ensemble_method in ensemble_results:
            ensemble_dict = ensemble_results[ensemble_method]

            ensemble_metrics = self.performance_analyzer.calculate_metrics(
                target=ensemble_dict['target'],
                predicted_labels=ensemble_dict['label'],
                predicted_probs=ensemble_dict['proba'],
                target_metrics=[
                    self.metric])

            ensemble_metric = ensemble_metrics[self.metric]
            if ensemble_metric > self.best_base_results[
                    'Base_metric'] and ensemble_metric > self.best_ensemble_metric:
                self.best_ensemble_metric = ensemble_metric
                top_ensemble_dict.update({ensemble_method: ensemble_metric})
        return dict(
            sorted(
                top_ensemble_dict.items(),
                key=lambda x: x[1],
                reverse=True))

    def agg_ensemble(self, modelling_results: dict = None,
                     single_mode: bool = False) -> dict:
        ensemble_dict = {}
        if single_mode:
            for strategy in self.ensemble_strategy:
                ensemble_dict.update({f'{strategy}': self._ensemble_by_method(
                    modelling_results, strategy=strategy)})
        else:
            for generator in modelling_results:
                ensemble_dict[generator] = {}
                self.generator = generator
                for launch in modelling_results[generator]:
                    ensemble_dict[generator].update(
                        {launch: modelling_results[generator][launch]['metrics']})

                for strategy in self.ensemble_strategy:
                    ensemble_dict[generator].update({strategy: self._ensemble_by_method(
                        modelling_results[generator], strategy=strategy)})
        return ensemble_dict

    def _ensemble_by_method(self, predictions, strategy):
        transformed_predictions = self._check_predictions(
            predictions, strategy_name=strategy)
        average_proba_predictions = self.ensemble_strategy_dict[strategy](
            transformed_predictions, axis=1)

        if average_proba_predictions.shape[1] == 1:
            average_proba_predictions = np.concatenate(
                [average_proba_predictions, 1 - average_proba_predictions], axis=1)

        label_predictions = np.argmax(average_proba_predictions, axis=1)
        try:
            target = self.test_target
            metrics = self.performance_analyzer.calculate_metrics(
                target=target,
                predicted_labels=label_predictions,
                predicted_probs=average_proba_predictions,
            )
        except KeyError:
            metrics = None
            target = None

        return {'target': target,
                'label': label_predictions,
                'proba': average_proba_predictions,
                'metrics': metrics}

    def _check_predictions(self, predictions, strategy_name):
        """Check if the predictions array has the correct size.

        Args:
            predictions: array of shape (n_samples, n_classifiers). The votes obtained by each classifier
            for each sample.
            strategy_name: str. The name of the strategy used to ensemble the predictions.

        Returns:
            predictions: array of shape (n_samples, n_classifiers). The votes obtained by each classifier
            for each sample.

        Raises:
            ValueError: if the array do not contain exactly 3 dimensions: [n_samples, n_classifiers, n_classes]

        """
        if strategy_name in self.strategy_exclude_list:
            return predictions

        if isinstance(predictions, dict):
            try:
                list_proba = []
                for model_preds in predictions:
                    proba_frame = predictions[model_preds]
                    if isinstance(proba_frame, np.ndarray):
                        list_proba.append(proba_frame)
                    else:
                        try:
                            list_proba.append(proba_frame.values)
                        except KeyError:
                            self.target = proba_frame['Target'].values
                            if 'Preds' in proba_frame.columns:
                                filter_col = ['Target', 'Preds']
                            else:
                                filter_col = ['Target', 'Predicted_labels']
                            proba_frame = proba_frame.loc[:, ~proba_frame.columns.isin(
                                filter_col)]
                            list_proba.append(proba_frame.values)
                return np.array(list_proba).transpose((1, 0, 2))
            except Exception as error:
                self.logger.error(f'Error in ensemble predictions: {error}')
                raise ValueError(
                    'predictions must contain 3 dimensions: ')
