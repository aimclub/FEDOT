from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Type, Union

from golem.core.optimisers.optimizer import GraphOptimizer

from fedot.api.main import Fedot
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.metrics_repository import MetricIDType
from fedot.core.repository.tasks import TaskParams


class DefaultParamValue:
    def __repr__(self):
        return '<default value>'  # Visible in the documentation.


DEFAULT_VALUE = DefaultParamValue()  # Mock value used to filter out unset parameters.


class FedotBuilder:
    """ An alternative FEDOT API version with optional attributes being documented
    and separated into groups by meaning.
    Each of the groups has corresponding setter method, named starting with :obj:`setup_*`.
    Use these methods to set corresponding API attributes:

    * :meth:`~FedotBuilder.setup_composition` -> general AutoML parameters

    * :meth:`~FedotBuilder.setup_parallelization` -> parameters of computational parallelization by CPU jobs

    * :meth:`~FedotBuilder.setup_output` -> parameters of outputs: logging, cache directories, etc.

    * :meth:`~FedotBuilder.setup_evolution` -> parameters of ML pipelines evolutionary optimization

    * :meth:`~FedotBuilder.setup_pipeline_structure` -> constrains on ML pipeline structure

    * :meth:`~FedotBuilder.setup_pipeline_evaluation` -> parameters of ML pipelines quality evaluation

    * :meth:`~FedotBuilder.setup_data_preprocessing` -> parameters of input data preprocessing

    After all demanded attributes are set, use :meth:`~FedotBuilder.build` to get a parametrized instance of
    :class:`~fedot.api.main.Fedot`.

    Examples:
        Example 1:

        .. literalinclude:: ../../../examples/simple/api_builder/classification_with_api_builder.py

        Example 2:

        .. literalinclude:: ../../../examples/simple/api_builder/multiple_ts_forecasting_tasks.py

    Args:
        problem: name of a modelling problem to solve.

            .. details:: Possible options:

                - ``classification`` -> for classification task
                - ``regression`` -> for regression task
                - ``ts_forecasting`` -> for time series forecasting task
"""

    def __init__(self, problem: str):
        self.api_params: Dict[Any, Any] = dict(problem=problem)

    def __update_params(self, **new_params):
        """ Saves all parameters set by user to the dictionary ``self.api_params``. """

        new_params = {k: v for k, v in new_params.items() if v != DEFAULT_VALUE}
        self.api_params.update(new_params)

    def setup_composition(
            self,
            timeout: Optional[float] = DEFAULT_VALUE,
            task_params: TaskParams = DEFAULT_VALUE,
            seed: int = DEFAULT_VALUE,
            preset: str = DEFAULT_VALUE,
            with_tuning: bool = DEFAULT_VALUE,
            use_meta_rules: bool = DEFAULT_VALUE,
    ) -> FedotBuilder:
        """ Sets general AutoML parameters.

        Args:
            timeout: time for model design (in minutes): ``None`` or ``-1`` means infinite time.

            task_params: additional parameters of a task.

            seed: value for a fixed random seed.

            preset: name of the preset for model building (e.g. ``best_quality``, ``fast_train``, ``gpu``).
                Default value is ``auto``.

                .. details:: Possible options:

                    - ``best_quality`` -> All models that are available for this data type and task are used
                    - ``fast_train`` -> Models that learn quickly. This includes preprocessing operations
                      (data operations) that only reduce the dimensionality of the data, but cannot increase it.
                      For example, there are no polynomial features and one-hot encoding operations
                    - ``stable`` -> The most reliable preset in which the most stable operations are included
                    - ``auto`` -> Automatically determine which preset should be used
                    - ``gpu`` -> Models that use GPU resources for computation
                    - ``ts`` -> A special preset with models for time series forecasting task
                    - ``automl`` -> A special preset with only AutoML libraries such as TPOT and H2O as operations

            with_tuning: flag for tuning hyperparameters of the final evolved
                :class:`~fedot.core.pipelines.pipeline.Pipeline`. Defaults to ``True``.

            use_meta_rules: indicates whether to change set parameters according to FEDOT meta rules.


        Returns:
            :class:`FedotBuilder` instance.
        """
        self.__update_params(
            timeout=timeout,
            task_params=task_params,
            seed=seed,
            preset=preset,
            with_tuning=with_tuning,
            use_meta_rules=use_meta_rules,
        )
        return self

    def setup_parallelization(
            self,
            n_jobs: int = DEFAULT_VALUE,
            parallelization_mode: str = DEFAULT_VALUE,
    ) -> FedotBuilder:
        """ Sets parameters of computational parallelization by CPU jobs.

        Args:
            n_jobs: num of `jobs` for parallelization (set to ``-1`` to use all cpu's). Defaults to ``-1``.
            parallelization_mode: type of evaluation for groups of individuals (``populational`` or
                ``sequential``). Default value is ``populational``.
        Returns:
            :class:`FedotBuilder` instance.
        """
        self.__update_params(
            n_jobs=n_jobs,
            parallelization_mode=parallelization_mode,
        )
        return self

    def setup_output(
            self,
            logging_level: int = DEFAULT_VALUE,
            show_progress: bool = DEFAULT_VALUE,
            keep_history: bool = DEFAULT_VALUE,
            history_dir: Optional[str] = DEFAULT_VALUE,
            cache_dir: Optional[str] = DEFAULT_VALUE,
    ) -> FedotBuilder:
        """ Sets parameters of outputs: logging, cache directories, etc.

        Args:
            logging_level: logging levels are the same as in
                `built-in logging library <https://docs.python.org/3/library/logging.html>`_.

                .. details:: Possible options:

                    - ``50`` -> critical
                    - ``40`` -> error
                    - ``30`` -> warning
                    - ``20`` -> info
                    - ``10`` -> debug
                    - ``0`` -> nonset

            show_progress: indicates whether to show progress using tqdm/tuner or not. Defaults to ``True``.

            keep_history: indicates if the framework should track evolutionary optimization history
                for possible further analysis. Defaults to ``True``.

            history_dir: relative or absolute path of a folder for composing history.
                By default, creates a folder named "FEDOT" in temporary system files of an OS.
                A relative path is relative to the default value.

            cache_dir: path to a directory containing cache files (if any cache is enabled).
                By default, creates a folder named "FEDOT" in temporary system files of an OS.

        Returns:
            :class:`FedotBuilder` instance.
        """
        self.__update_params(
            logging_level=logging_level,
            show_progress=show_progress,
            keep_history=keep_history,
            history_dir=history_dir,
            cache_dir=cache_dir,
        )
        return self

    def setup_evolution(
            self,
            initial_assumption: Union[Pipeline, List[Pipeline]] = DEFAULT_VALUE,
            num_of_generations: Optional[int] = DEFAULT_VALUE,
            early_stopping_iterations: int = DEFAULT_VALUE,
            early_stopping_timeout: int = DEFAULT_VALUE,
            pop_size: int = DEFAULT_VALUE,
            keep_n_best: int = DEFAULT_VALUE,
            genetic_scheme: str = DEFAULT_VALUE,
            use_pipelines_cache: bool = DEFAULT_VALUE,
            optimizer: Optional[Type[GraphOptimizer]] = DEFAULT_VALUE,
    ) -> FedotBuilder:
        """ Sets parameters of ML pipelines evolutionary optimization.

        Args:
            initial_assumption: initial assumption(s) for composer.
                Can be either a single :class:`Pipeline` or sequence of ones. Default values are task-specific and
                selected by the method
                :meth:`~fedot.api.api_utils.assumptions.task_assumptions.TaskAssumptions.for_task`.

            early_stopping_iterations: composer will stop after ``n`` generation without improving

            num_of_generations: number of evolutionary generations for composer. Defaults to ``None`` - no limit.

            early_stopping_iterations: composer will stop after ``n`` generation without improving.

            early_stopping_timeout: stagnation timeout in minutes: composer will stop after ``n`` minutes
                without improving. Defaults to ``10``.

            pop_size: size of population (generation) during composing. Defaults to ``20``.

            keep_n_best: number of the best individuals in generation that survive during the evolution.
                 Defaults to ``1``.

            genetic_scheme: name of the genetic scheme. Defaults to ``steady_state``.

            use_pipelines_cache: indicates whether to use pipeline structures caching. Defaults to ``True``.

            optimizer: inherit from :class:`golem.core.optimisers.optimizer.GraphOptimizer`
                to specify a custom optimizer.
                Default optimizer is :class:`golem.core.optimisers.genetic.gp_optimizer.EvoGraphOptimizer`.
                See the :doc:`example </advanced/external_optimizer>`.

        Returns:
            :class:`FedotBuilder` instance.
        """
        self.__update_params(
            initial_assumption=initial_assumption,
            num_of_generations=num_of_generations,
            early_stopping_iterations=early_stopping_iterations,
            early_stopping_timeout=early_stopping_timeout,
            pop_size=pop_size,
            keep_n_best=keep_n_best,
            genetic_scheme=genetic_scheme,
            use_pipelines_cache=use_pipelines_cache,
            optimizer=optimizer,
        )
        return self

    def setup_pipeline_structure(
            self,
            available_operations: List[str] = DEFAULT_VALUE,
            max_depth: int = DEFAULT_VALUE,
            max_arity: int = DEFAULT_VALUE,
    ) -> FedotBuilder:
        """ Sets constrains on ML pipeline structure.

        Args:
            available_operations: list of model names to use. Pick the names according to operations repository.

                .. details:: All options:

                    - ``adareg`` -> AdaBoost Regressor
                    - ``ar`` -> AutoRegression
                    - ``arima`` -> ARIMA
                    - ``cgru`` -> Convolutional Gated Recurrent Unit
                    - ``bernb`` -> Naive Bayes Classifier (multivariate Bernoulli)
                    - ``catboost`` -> Catboost Classifier
                    - ``catboostreg`` -> Catboost Regressor
                    - ``dt`` -> Decision Tree Classifier
                    - ``dtreg`` -> Decision Tree Regressor
                    - ``gbr`` -> Gradient Boosting Regressor
                    - ``kmeans`` -> K-Means clustering
                    - ``knn`` -> K-nearest neighbors Classifier
                    - ``knnreg`` -> K-nearest neighbors Regressor
                    - ``lasso`` -> Lasso Linear Regressor
                    - ``lda`` -> Linear Discriminant Analysis
                    - ``lgbm`` -> Light Gradient Boosting Machine Classifier
                    - ``lgbmreg`` -> Light Gradient Boosting Machine Regressor
                    - ``linear`` -> Linear Regression Regressor
                    - ``logit`` -> Logistic Regression Classifier
                    - ``mlp`` -> Multi-layer Perceptron Classifier
                    - ``multinb`` -> Naive Bayes Classifier (multinomial)
                    - ``qda`` -> Quadratic Discriminant Analysis
                    - ``rf`` -> Random Forest Classifier
                    - ``rfr`` -> Random Forest Regressor
                    - ``ridge`` -> Ridge Linear Regressor
                    - ``polyfit`` -> Polynomial fitter
                    - ``sgdr`` -> Stochastic Gradient Descent Regressor
                    - ``stl_arima`` -> STL Decomposition with ARIMA
                    - ``glm`` -> Generalized Linear Models
                    - ``ets`` -> Exponential Smoothing
                    - ``locf`` -> Last Observation Carried Forward
                    - ``ts_naive_average`` -> Naive Average
                    - ``svc`` -> Support Vector Classifier
                    - ``svr`` -> Linear Support Vector Regressor
                    - ``treg`` -> Extra Trees Regressor
                    - ``xgboost`` -> Extreme Gradient Boosting Classifier
                    - ``xgbreg`` -> Extreme Gradient Boosting Regressor
                    - ``cnn`` -> Convolutional Neural Network
                    - ``scaling`` -> Scaling
                    - ``normalization`` -> Normalization
                    - ``simple_imputation`` -> Imputation
                    - ``pca`` -> Principal Component Analysis
                    - ``kernel_pca`` -> Kernel Principal Component Analysis
                    - ``fast_ica`` -> Independent Component Analysis
                    - ``poly_features`` -> Polynomial Features
                    - ``one_hot_encoding`` -> One-Hot Encoder
                    - ``label_encoding`` -> Label Encoder
                    - ``rfe_lin_reg`` -> Linear Regression Recursive Feature Elimination
                    - ``rfe_non_lin_reg`` -> Decision Tree Recursive Feature Elimination
                    - ``rfe_lin_class`` -> Logistic Regression Recursive Feature Elimination
                    - ``rfe_non_lin_class`` -> Decision Tree Recursive Feature Elimination
                    - ``isolation_forest_reg`` -> Regression Isolation Forest
                    - ``isolation_forest_class`` -> Classification Isolation Forest
                    - ``decompose`` -> Regression Decomposition
                    - ``class_decompose`` -> Classification Decomposition
                    - ``resample`` -> Resample features
                    - ``ransac_lin_reg`` -> Regression Random Sample Consensus
                    - ``ransac_non_lin_reg`` -> Decision Tree Random Sample Consensus
                    - ``cntvect`` -> Count Vectorizer
                    - ``text_clean`` -> Lemmatization and Stemming
                    - ``tfidf`` -> TF-IDF Vectorizer
                    - ``word2vec_pretrained`` -> Word2Vec
                    - ``lagged`` -> Lagged Transformation
                    - ``sparse_lagged`` -> Sparse Lagged Transformation
                    - ``smoothing`` -> Smoothing Transformation
                    - ``gaussian_filter`` -> Gaussian Filter Transformation
                    - ``diff_filter`` -> Derivative Filter Transformation
                    - ``cut`` -> Cut Transformation
                    - ``exog_ts`` -> Exogeneus Transformation
                    - ``topological_features`` -> Topological features

                .. details:: Available for composing tabular models:

                    - ``catboost`` -> Catboost Classifier
                    - ``catboostreg`` -> Catboost Regressor
                    - ``knn`` -> K-nearest neighbors Classifier
                    - ``knnreg`` -> K-nearest neighbors Regressor
                    - ``lgbm`` -> Light Gradient Boosting Machine Classifier
                    - ``lgbmreg`` -> Light Gradient Boosting Machine Regressor
                    - ``logit`` -> Logistic Regression Classifier
                    - ``rf`` -> Random Forest Classifier
                    - ``rfr`` -> Random Forest Regressor
                    - ``ridge`` -> Ridge Linear Regressor
                    - ``treg`` -> Extra Trees Regressor
                    - ``xgboost`` -> Extreme Gradient Boosting Classifier
                    - ``xgbreg`` -> Extreme Gradient Boosting Regressor

            max_depth: max depth of a pipeline. Defaults to ``6``.

            max_arity: max arity of a pipeline nodes. Defaults to ``3``.

        Returns:
            :class:`FedotBuilder` instance.
        """
        self.__update_params(
            available_operations=available_operations,
            max_depth=max_depth,
            max_arity=max_arity,
        )
        return self

    def setup_pipeline_evaluation(
            self,
            metric: Union[MetricIDType, Sequence[MetricIDType]] = DEFAULT_VALUE,
            cv_folds: int = DEFAULT_VALUE,
            max_pipeline_fit_time: Optional[int] = DEFAULT_VALUE,
            collect_intermediate_metric: bool = DEFAULT_VALUE,
    ) -> FedotBuilder:
        """ Sets parameters of ML pipelines quality evaluation.

        Args:
            metric:
                metric for quality calculation during composing, also is used for tuning if ``with_tuning=True``.

                .. details:: Default value depends on a given task:

                    - ``roc_auc_pen`` -> for classification
                    - ``rmse`` -> for regression & time series forecasting

                .. details:: Available metrics are listed in the following enumerations:

                    - classification -> \
                        :class:`~fedot.core.repository.metrics_repository.ClassificationMetricsEnum`
                    - regression -> \
                        :class:`~fedot.core.repository.metrics_repository.RegressionMetricsEnum`
                    - time series forcasting -> \
                        :class:`~fedot.core.repository.metrics_repository.TimeSeriesForecastingMetricsEnum`
                    - pipeline complexity (task-independent) -> \
                        :class:`~fedot.core.repository.metrics_repository.ComplexityMetricsEnum`

            cv_folds: number of folds for cross-validation.

                .. details:: Default value depends on a given ``problem``:

                    - ``5`` -> for classification and regression tasks
                    - ``3`` -> for time series forecasting task

            max_pipeline_fit_time: time constraint for operation fitting (in minutes).
                Once the limit is reached, a candidate pipeline will be dropped. Defaults to ``None`` - no limit.

            collect_intermediate_metric: save metrics for intermediate (non-root) nodes in composed
                :class:`Pipeline`.

        Returns:
            :class:`FedotBuilder` instance.
        """
        self.__update_params(
            metric=metric,
            cv_folds=cv_folds,
            max_pipeline_fit_time=max_pipeline_fit_time,
            collect_intermediate_metric=collect_intermediate_metric,
        )
        return self

    def setup_data_preprocessing(
            self,
            safe_mode: bool = DEFAULT_VALUE,
            use_input_preprocessing: bool = DEFAULT_VALUE,
            use_preprocessing_cache: bool = DEFAULT_VALUE,
            use_auto_preprocessing: bool = DEFAULT_VALUE,
    ) -> FedotBuilder:
        """ Sets parameters of input data preprocessing.

        Args:
            safe_mode: if set ``True`` it will cut large datasets to prevent memory overflow and use label encoder
                instead of one-hot encoder if summary cardinality of categorical features is high.
                Default value is ``False``.

            use_input_preprocessing: indicates whether to do preprocessing of further given data.
                Defaults to ``True``.

            use_preprocessing_cache: bool indicating whether to use optional preprocessors caching.
                Defaults to ``True``.

        Returns:
            :class:`FedotBuilder` instance.
        """
        self.__update_params(
            safe_mode=safe_mode,
            use_input_preprocessing=use_input_preprocessing,
            use_preprocessing_cache=use_preprocessing_cache,
            use_auto_preprocessing=use_auto_preprocessing,
        )
        return self

    def build(self) -> Fedot:
        """ Initializes an instance of :class:`~fedot.api.main.Fedot` with accumulated parameters.

        Returns:
            :class:`~fedot.api.main.Fedot` instance.
        """
        return Fedot(**self.api_params)
