import numpy as np
from fedot.confidence_intervals.tuners import quantile_loss_tuners
from fedot.confidence_intervals.utils import pipeline_simple_structure
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.confidence_intervals.utils import quantile_array,median_array,absolute_array, truncated_mean_array,mean_array
from fedot.confidence_intervals.visualization import plot_confidence_intervals
from fedot.core.repository.tasks import Task,TaskTypesEnum,TsForecastingParams

class ConfidenceIntervals:
    def __init__(self, model, train_input):
        self.model = model
        self.train_input = train_input
        
        self.ts = model.train_data.features
        self.horizon_fit = model.params.task_params.forecast_length
        self.model_forecast = model.forecast(horizon = self.horizon_fit)
        self.model_pipeline = model.current_pipeline
        self.task = Task(TaskTypesEnum.ts_forecasting,TsForecastingParams(forecast_length=self.horizon_fit))
        
        self.up_predictions = None
        self.low_predictions = None      
        self.up_int = None
        self.low_int = None
        self.horizon_test = None
        
    def fit(self, up_quantile,low_quantile, horizon = None,
            low_tuner = None, up_tuner = None,number_models = 10,pipelines_show = True):
        
        
        tuners = quantile_loss_tuners(up_quantile = up_quantile, 
                                     low_quantile = low_quantile,
                                     task = self.task,
                                     train_input = self.train_input,
                                     validations_blocks = 2,n_jobs = -1,show_progress = True)
        if up_tuner == None:
            up_tuner = tuners['up_tuner']
        if low_tuner == None:
            low_tuner = tuners['low_tuner']
        
        fits = []
        pipeline_dicts = []
        for ind in self.model.history.individuals[-2]:
            pipeline = pipeline_simple_structure(PipelineAdapter().restore(ind.graph))
            if pipeline not in pipeline_dicts:
                pipeline_dicts.append(pipeline)
                fits.append(ind.fitness.value)
        fits = np.sort(np.array(fits))

        up_predictions = []
        low_predictions = []
        s = 1
        if number_models == 'max':
            number_iterations = len(fits)-1
            print(f'{number_iterations+1} pipelines will be used during fitting')
        elif number_models >len(fits):
            number_iterations = len(fits)-1
            print(f'WARNING! Number models for training is bigger than number of different pipelines. All {number_iterations+1} different pipelines are used for training.')
        else:
            number_iterations = number_models-1
    
        
        for ind in self.model.history.individuals[-2]: 
            if ind.fitness.value <= fits[number_iterations]:           
                p = pipeline_simple_structure(PipelineAdapter().restore(ind.graph))
                if p in pipeline_dicts:
                    pipeline_dicts.remove(p)
                    print(f'Fitting pipeline №{s}')
                    s+=1
                
                    pipeline = PipelineAdapter().restore(ind.graph)
                    if pipelines_show:
                        pipeline.show()
                
                    tuned_pipeline = up_tuner.tune(pipeline)
                    tuned_pipeline.fit(self.train_input)
                    self.model.current_pipeline = tuned_pipeline
                    self.model.current_pipeline.show()
                
                    if horizon == None:
                        preds = self.model.forecast()
                    else:
                        preds = self.model.forecast(horizon=horizon)
                    up_predictions.append(preds)
                    del tuned_pipeline
                    
                
                #tuning lower confidence interval
                    tuned_pipeline = low_tuner.tune(pipeline)
                    tuned_pipeline.fit(self.train_input)
                    self.model.current_pipeline = tuned_pipeline
                    if horizon ==None:
                        preds = self.model.forecast()
                    else:
                        preds = self.model.forecast(horizon=horizon)
                    low_predictions.append(preds)
                    del tuned_pipeline
                    
        self.up_predictions = up_predictions
        self.low_predictions = low_predictions
        self.model.current_pipeline = self.model_pipeline
        self.horizon_test = horizon
        
        return {'up_predictions':up_predictions, 'low_predictions':low_predictions}  
    
    def forecast(self,regim = 'quantile'):
        if self.up_predictions == None:
            print('Error! Confidence intervals are not computed!')
        
        forecast = self.model.forecast(horizon = self.horizon_test)
        if regim == 'quantile':
            up_res= quantile_array(quantile = 0.1, arrays = self.up_predictions)['up']
            low_res= quantile_array(quantile = 0.1, arrays = self.low_predictions)['low']
        if regim == 'mean':
            up_res = mean_array(arrays = self.up_predictions)
            low_res = mean_array(arrays = self.low_predictions)
        if regim == 'median':
            up_res = median_array(arrays = self.up_predictions)
            low_res = median_array(arrays = self.low_predictions)
        if regim == 'absolute_bounds':
            up_res = absolute_array(arrays = self.up_predictions)['up']
            low_res = absolute_array(arrays = self.low_predictions)['low']
        if regim == 'truncated_mean':
            up_res = truncated_mean_array(arrays = self.up_predictions)
            low_res = truncated_mean_array(arrays = self.low_predictions)
    
        up = np.maximum(up_res,forecast)  
        low = np.minimum(low_res,forecast)
        self.up_int = up
        self.low_int = low
        
        return {'up':up,'low':low}
        
        
          
    def plot(self,
             up_int = True,
             low_int = True,
             show_forecast = True,
             history = True,
             up_train = True,
             low_train = True,
             ts_test = None):

        plot_confidence_intervals(horizon=len(self.up_int),
                                  up_predictions = self.up_predictions,
                                  low_predictions = self.low_predictions,
                                  model_forecast = self.model.forecast(horizon = self.horizon_test),
                                  up = self.up_int,
                                  low = self.low_int,
                                  ts = self.ts,
                                  up_int = up_int,
                                  low_int = low_int,
                                  forecast = show_forecast,
                                  history = history,
                                  up_train = up_train,
                                  low_train = low_train,
                                  ts_test = ts_test)