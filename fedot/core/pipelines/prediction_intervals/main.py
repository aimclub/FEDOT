import numpy as np
from functools import partial


from golem.core.log import default_log, Log
from fedot.api.main import Fedot
from fedot.core.data.data import InputData

from fedot.core.pipelines.prediction_intervals.utils import compute_prediction_intervals, \
    get_base_quantiles, check_init_params, get_last_generations
from fedot.core.pipelines.prediction_intervals.visualization import plot_prediction_intervals, \
    _plot_prediction_intervals
from fedot.core.pipelines.prediction_intervals.solvers.best_pipelines_quantiles import solver_best_pipelines_quantiles
from fedot.core.pipelines.prediction_intervals.solvers.last_generation_quantile_loss import solver_last_generation_ql
from fedot.core.pipelines.prediction_intervals.solvers.mutation_of_best_pipeline import solver_mutation_of_best_pipeline
from fedot.core.pipelines.prediction_intervals.params import PredictionIntervalsParams


class PredictionIntervals:
    """Class for building prediction intervals based on Fedot functionality.

       Args:
           model: a fitted Fedot class object for times series forecasting task
           horizon: horizon for ts forecasting
           nominal_error: nominal error for prediction intervals task
           method: general method for creating forecasts. Possible options:

                - 'last_generation_ql' -> a number of pipelines from the last generation is fitted using
                   quantile loss/pinball loss metric.
                - 'best_pipelines_quantiles' -> prediction intervals are computed based on predictions of all
                   pipelines in last generation
                - 'mutation_of_best_pipeline' -> prediction intervals are built relying on mutations of the final
                   pipeline

           params: params for building prediction intervals.
    """

    def __init__(self,
                 model: Fedot,
                 horizon: int = None,
                 nominal_error: float = 0.1,
                 method: str = 'mutation_of_best_pipeline',
                 params: PredictionIntervalsParams = PredictionIntervalsParams()):

        # check whether given Fedot class object is fitted and argument 'method' is written correctly
        check_init_params(model, method)

        last_generations = get_last_generations(model)
        self.generation = last_generations['last_generation']
        self.best_ind = last_generations['final_choice']
        self.ts = model.train_data.features
        
        self.horizon = horizon
        self.nominal_error = nominal_error
        self.method = method
        self.model_forecast = model.forecast(horizon=horizon)
        self.best_pipeline = model.current_pipeline

        # setup logger
        Log().reset_logging_level(params.logging_level)
        self.logger = default_log(prefix='PredictionIntervals')

        # arrays of auxilary forecasts to build prediction intervals
        self.up_predictions = None
        self.low_predictions = None
        self.all_predictions = None

        # prediction intervals
        self.up_int = None
        self.low_int = None

        # base quantiles
        self.base_quantiles = None

        # flags whether PredictionIntervals instance is fitted/forecasted
        self.is_fitted = False
        self.is_forecasted = False

        # flag whether base quantiles are computed
        self.base_quantiles_are_computed = False

        # initialize solver for building prediction intervals
        if self.method == 'mutation_of_best_pipeline':
            self.solver = partial(solver_mutation_of_best_pipeline,
                                  ind=self.best_ind,
                                  horizon=self.horizon,
                                  forecast=self.model_forecast,
                                  number_mutations=params.number_mutations,
                                  mutations_choice=params.mutations_choice,
                                  n_jobs=params.n_jobs,
                                  show_progress=params.show_progress,
                                  discard_inapropriate_pipelines=params.mutations_discard_inapropriate_pipelines,
                                  keep_percentage=params.mutation_keep_percentage,
                                  logger=self.logger)

        elif self.method == 'best_pipelines_quantiles':
            self.solver = partial(solver_best_pipelines_quantiles,
                                  generation=self.generation,
                                  horizon=self.horizon,
                                  number_models=params.bpq_number_models,
                                  show_progress=params.show_progress,
                                  logger=self.logger)

        elif self.method == 'last_generation_ql':
            self.solver = partial(solver_last_generation_ql,
                                  generation=self.generation,
                                  horizon=self.horizon,
                                  nominal_error=self.nominal_error,
                                  number_models=params.ql_number_models,
                                  iterations=params.ql_tuner_iterations,
                                  minutes=params.ql_tuner_minutes,
                                  n_jobs=params.n_jobs,
                                  show_progress=params.show_progress,
                                  validation_blocks=params.ql_tuner_validation_blocks,
                                  up_tuner=params.ql_up_tuner,
                                  low_tuner=params.ql_low_tuner,
                                  logger=self.logger)


    regime_up = {'quantile': 'quantile_up', 'mean': 'mean', 'median': 'median', 'absolute_bounds': 'max'}
    regime_low = {'quantile': 'quantile_low', 'mean': 'mean', 'median': 'median', 'absolute_bounds': 'min'}


    def fit(self, train_input: InputData):
        """This method creates several np.arrays that will be used in method 'forecast' to build prediction intervals.

        Fitting process rans by self.solver initialized in method '__init__'. According to the solver several pipelines
        are generated and their forecasts are transfered then in method 'forecast'.

        Args:
            train_input: train data used for training the model.
        """
        x = self.solver(train_input)

        if self.method == 'last_generation_ql':
            self.up_predictions = x['up_predictions']
            self.low_predictions = x['low_predictions']
        else:
            self.all_predictions = x

        self.is_fitted = True

    def forecast(self, regime: str = 'quantile'):
        """This method builds prediction intervals based on the output of method 'fit'.

           Args:
               regime (str): a way to compute prediction intervals if argument 'method' is 'last_generation_ql'.

           Returns:
               dictionary of upper and low prediction intervals.
        """
        if not self.is_fitted:
            self.logger.critical('PredictionIntervals instance is not fitted! Fit the instance first.')

        if self.method == 'last_generation_ql':
            quantiles_up = compute_prediction_intervals(self.up_predictions, nominal_error=self.nominal_error)
            quantiles_low = compute_prediction_intervals(self.low_predictions, nominal_error=self.nominal_error)

            up_int = quantiles_up[self.regime_up[regime]]
            low_int = quantiles_low[self.regime_low[regime]]

        elif self.method in ['best_pipelines_quantiles', 'mutation_of_best_pipeline']:

            quantiles = compute_prediction_intervals(self.all_predictions, nominal_error=self.nominal_error)
            up_int = quantiles['quantile_up']
            low_int = quantiles['quantile_low']

        self.up_int = np.maximum(up_int, self.model_forecast)
        self.low_int = np.minimum(low_int, self.model_forecast)

        self.is_forecasted = True

        return {'up_int': self.up_int, 'low_int': self.low_int}


    def get_base_quantiles(self, train_input: InputData):
        """Method to get quantiles based on predictions of final pipeline over train_data.

        Args:
            train_input: train data.

        Returns:
            dictionary consisting of upper and low quantiles computed for model forecast residuals over train ts.
        """
        base_quantiles = get_base_quantiles(train_input,
                                            pipeline=self.best_pipeline,
                                            nominal_error=self.nominal_error)

        self.base_quantiles = {'up': self.model_forecast + base_quantiles['up'],
                               'low': self.model_forecast + base_quantiles['low']}
        self.base_quantiles_are_computed = True

        return self.base_quantiles


    def plot(self,
             show_history: bool = True,
             show_forecast: bool = True,
             ts_test: np.array = None):
        """Method for plotting obtained prediction intervals, model forecast and test data."""

        if self.is_forecasted is False:
            self.logger.critical('Prediction intervals are not built! Use fit and then forecast methods first.')

        plot_prediction_intervals(model_forecast=self.model_forecast,
                                  up_int=self.up_int,
                                  low_int=self.low_int,
                                  ts=self.ts,
                                  show_history=show_history,
                                  show_forecast=show_forecast,
                                  ts_test=ts_test,
                                  labels='pred_ints')


    def plot_base_quantiles(self,
                           show_history: bool = True,
                           show_forecast: bool = True,
                           ts_test: np.array = None):
        """Method for plotting prediction intervals built on base quantiles, model forecast and test data."""


        if self.base_quantiles_are_computed is False:
            self.logger.critical('Base quantiles are not computed! Use get_base_quantiles method first.')

        plot_prediction_intervals(model_forecast=self.model_forecast,
                                  up_int=self.base_quantiles['up'],
                                  low_int=self.base_quantiles['low'],
                                  ts=self.ts,
                                  show_history=show_history,
                                  show_forecast=show_forecast,
                                  ts_test=ts_test,
                                  labels='base_quantiles')


    def _plot(self,
              show_up_int=True,
              show_low_int=True,
              show_forecast=True,
              show_history=True,
              show_up_train=True,
              show_low_train=True,
              show_train=True,
              ts_test: np.array = None):
        """ Old method for plotting prediction intervals, train and test data. Used for developing, will be removed."""

        _plot_prediction_intervals(horizon=len(self.model_forecast),
                                   up_predictions=self.up_predictions,
                                   low_predictions=self.low_predictions,
                                   predictions=self.all_predictions,
                                   model_forecast=self.model_forecast,
                                   up_int=self.up_int,
                                   low_int=self.low_int,
                                   ts=self.ts,
                                   show_up_int=show_up_int,
                                   show_low_int=show_low_int,
                                   show_forecast=show_forecast,
                                   show_history=show_history,
                                   show_up_train=show_up_train,
                                   show_low_train=show_low_train,
                                   show_train=show_train,
                                   ts_test=ts_test)
