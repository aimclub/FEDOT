import os
import datetime
import warnings

import numpy as np
import pandas as pd

from fedot.core.composer.gp_composer.gp_composer import \
    GPComposerBuilder, GPComposerRequirements
from fedot.core.composer.gp_composer.specific_operators import parameter_change_mutation
from fedot.core.data.data import InputData
from fedot.core.optimisers.gp_comp.gp_optimiser import GPGraphOptimiserParameters
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import \
    MetricsRepository, RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.utilities.ts_gapfilling import SimpleGapFiller
from examples.ts_forecasting_composing import get_source_pipeline
from methods.validation_and_metrics import validate
warnings.filterwarnings('ignore')


class ComposerGapFiller(SimpleGapFiller):
    """
    Class used for filling in the gaps in time series

    :param gap_value: value, which mask gap elements in array
    :param chain: TsForecastingChain object for filling in the gaps
    """

    def __init__(self, gap_value, chain):
        super().__init__(gap_value)
        self.chain = chain

    def forward_inverse_filling(self, input_data):
        """
        Method fills in the gaps in the input array using forward and inverse
        directions of predictions

        :param input_data: data with gaps to filling in the gaps in it
        :return: array without gaps
        """

        output_data = np.array(input_data)

        # Gap indices
        gap_list = np.ravel(np.argwhere(output_data == self.gap_value))
        new_gap_list = self._parse_gap_ids(gap_list)

        # Iterately fill in the gaps in the time series
        for batch_index in range(len(new_gap_list)):

            preds = []
            weights = []
            # Two predictions are generated for each gap - forward and backward
            for direction_function in [self._forward, self._inverse]:

                weights_list, predicted_list = direction_function(output_data,
                                                                  batch_index,
                                                                  new_gap_list)
                weights.append(weights_list)
                preds.append(predicted_list)

            preds = np.array(preds)
            weights = np.array(weights)
            result = np.average(preds, axis=0, weights=weights)

            gap = new_gap_list[batch_index]
            # Replace gaps in an array with predicted values
            output_data[gap] = result

        return output_data

    def forward_filling(self, input_data):
        """
        Method fills in the gaps in the input array using chain with only
        forward direction (i.e. time series forecasting)

        :param input_data: data with gaps to filling in the gaps in it
        :return: array without gaps
        """

        output_data = np.array(input_data)

        # Gap indices
        gap_list = np.ravel(np.argwhere(output_data == self.gap_value))
        new_gap_list = self._parse_gap_ids(gap_list)

        # Iterately fill in the gaps in the time series
        for gap in new_gap_list:
            # The entire time series is used for training until the gap
            timeseries_train_part = output_data[:gap[0]]

            # Adaptive prediction interval length
            len_gap = len(gap)

            # Chain for the task of filling in gaps
            predicted = self.__chain_fit_predict(timeseries_train_part,
                                                 len_gap)

            # Replace gaps in an array with predicted values
            output_data[gap] = predicted
        return output_data

    def _forward(self, timeseries_data, batch_index, new_gap_list):
        """
        The time series method makes a forward forecast based on the part
        of the time series that is located to the left of the gap.

        :param timeseries_data: one-dimensional array of a time series
        :param batch_index: index of the interval (batch) with a gap
        :param new_gap_list: array with nested lists of gap indexes

        :return weights_list: numpy array with prediction weights for
        averaging
        :return predicted_values: numpy array with predicted values in the
        gap
        """

        gap = new_gap_list[batch_index]
        timeseries_train_part = timeseries_data[:gap[0]]

        # Adaptive prediction interval length
        len_gap = len(gap)
        predicted_values = self.__chain_fit_predict(timeseries_train_part,
                                                    len_gap)
        weights_list = np.arange(len_gap, 0, -1)
        return weights_list, predicted_values

    def _inverse(self, timeseries_data, batch_index, new_gap_list):
        """
        The time series method makes an inverse forecast based on the part
        of the time series that is located to the right of the gap.

        :param timeseries_data: one-dimensional array of a time series
        :param batch_index: index of the interval (batch) with a gap
        :param new_gap_list: array with nested lists of gap indexes

        :return weights_list: numpy array with prediction weights for
        averaging
        :return predicted_values: numpy array with predicted values in the
        gap
        """

        gap = new_gap_list[batch_index]

        # If the interval with a gap is the last one in the array
        if batch_index == len(new_gap_list) - 1:
            timeseries_train_part = timeseries_data[(gap[-1] + 1):]
        else:
            next_gap = new_gap_list[batch_index + 1]
            timeseries_train_part = timeseries_data[(gap[-1] + 1):next_gap[0]]
        timeseries_train_part = np.flip(timeseries_train_part)

        # Adaptive prediction interval length
        len_gap = len(gap)

        predicted_values = self.__chain_fit_predict(timeseries_train_part,
                                                    len_gap)

        predicted_values = np.flip(predicted_values)
        weights_list = np.arange(1, (len_gap + 1), 1)
        weights_list = weights_list*0.5
        return weights_list, predicted_values

    def __chain_fit_predict(self, timeseries_train: np.array, len_gap: int):
        """
        The method makes a prediction as a sequence of elements based on a
        training sample. There are two main parts: fit model and predict.

        :param timeseries_train: part of the time series for training the model
        :param len_gap: number of elements in the gap
        :return: array without gaps
        """

        task = Task(TaskTypesEnum.ts_forecasting,
                    TsForecastingParams(forecast_length=len_gap))

        # Prepare data for train and predict
        input_train = InputData(idx=np.arange(0, len(timeseries_train)),
                                features=timeseries_train, target=timeseries_train,
                                task=task, data_type=DataTypesEnum.ts)

        start_forecast = len(timeseries_train)
        end_forecast = start_forecast + len_gap
        input_predict = InputData(idx=np.arange(start_forecast, end_forecast),
                                  features=timeseries_train, target=None,
                                  task=task, data_type=DataTypesEnum.ts)

        primary_operations = ['linear', 'ridge', 'lasso', 'dtreg', 'knnreg']

        secondary_operations = ['linear', 'ridge', 'lasso', 'rfr', 'dtreg',
                                'knnreg', 'svr']

        composer_requirements = GPComposerRequirements(
            primary=primary_operations,
            secondary=secondary_operations, max_arity=3,
            max_depth=8, pop_size=10, num_of_generations=2,
            crossover_prob=0.8, mutation_prob=0.8,
            timeout=datetime.timedelta(minutes=10))

        mutation_types = [parameter_change_mutation, MutationTypesEnum.simple,
                          MutationTypesEnum.reduce]
        optimiser_parameters = GPGraphOptimiserParameters(mutation_types=mutation_types)

        metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.MAE)
        builder = GPComposerBuilder(task=task). \
            with_optimiser_parameters(optimiser_parameters). \
            with_requirements(composer_requirements). \
            with_metrics(metric_function).with_initial_pipeline(self.chain)
        composer = builder.build()

        obtained_pipeline = composer.compose_pipeline(data=input_train, is_visualise=False)
        obtained_pipeline.print_structure()

        predicted_values = obtained_pipeline.predict(input_predict)
        return predicted_values


def run_fedot_composer(folder_to_save, files_list,
                       columns_with_gap, file_with_results,
                       vis = False):
    """
    The function starts the algorithm of gap-filling

    :param folder_to_save: where to save csv files with filled gaps
    :param files_list: list with file name, which will be processed
    :param columns_with_gap: list with names of columns with gaps
    :param file_with_results: file with final report with metrics
    :param vis: is there a need to make visualisations
    """

    # Create folder if it doesnt exists
    if os.path.isdir(folder_to_save) == False:
        os.makedirs(folder_to_save)

    mapes = []
    for file_id, file in enumerate(files_list):
        data = pd.read_csv(os.path.join('..', 'data', file))
        data['Date'] = pd.to_datetime(data['Date'])
        dataframe = data.copy()

        # Creating the dataframe
        mini_dataframe = pd.DataFrame({'File': [file]*6,
                                       'Metric': ['MAE', 'RMSE', 'MedAE',
                                                  'MAPE', 'Min gap value',
                                                  'Max gap value']})

        # For every gap series
        for column_with_gap in columns_with_gap:
            print(f'File - {file}, column with gaps - {column_with_gap}')
            array_with_gaps = np.array(data[column_with_gap])

            # Initial assumption
            init_pipeline = get_source_pipeline()
            gapfiller = ComposerGapFiller(gap_value=-100.0,
                                          chain=init_pipeline)
            withoutgap_arr = gapfiller.forward_filling(array_with_gaps)

            # Impute time series with new one
            dataframe[column_with_gap] = withoutgap_arr
            min_val, max_val, mae, rmse, medianae, mape = validate(parameter='Height',
                                                                   mask=column_with_gap,
                                                                   data=data,
                                                                   withoutgap_arr=withoutgap_arr,
                                                                   vis=vis)

            mini_dataframe[column_with_gap] = [mae, rmse, medianae, mape, min_val, max_val]
            mapes.append(mape)

            # Save resulting file
            save_path = os.path.join(folder_to_save, file)
            dataframe.to_csv(save_path)

        print(mini_dataframe)
        print('\n')

        if file_id == 0:
            main_dataframe = mini_dataframe
        else:
            frames = [main_dataframe, mini_dataframe]
            main_dataframe = pd.concat(frames)

    mapes = np.array(mapes)
    print(f'Mean MAPE value - {np.mean(mapes):.4f}')

    path_to_save = os.path.dirname(os.path.abspath(file_with_results))
    if os.path.isdir(path_to_save) == False:
        os.makedirs(path_to_save)
    main_dataframe.to_csv(file_with_results, index=False)



# Run the comopser example
folder_to_save = '../data/fedot_composer'
files_list = ['Synthetic.csv', 'Sea_hour.csv', 'Sea_10_240.csv', 'Temperature.csv', 'Traffic.csv']
columns_with_gap = ['gap', 'gap_center']
file_with_results = '../data/reports/fedot_composer_report.csv'

if __name__ == '__main__':
    run_fedot_composer(folder_to_save, files_list,
                       columns_with_gap, file_with_results, vis=True)