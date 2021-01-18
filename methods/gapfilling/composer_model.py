import datetime

import os
import pandas as pd
import numpy as np
from fedot.utilities.ts_gapfilling import SimpleGapFiller
from methods.validation_and_metrics import *
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.ts_chain import TsForecastingChain
from fedot.core.composer.gp_composer.gp_composer import \
    GPComposerBuilder, GPComposerRequirements
from fedot.core.composer.visualisation import ComposerVisualiser
from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import \
    MetricsRepository, RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import project_root

class ComposerGapFiller(SimpleGapFiller):
    """
    Class used for filling in the gaps in time series

    :param gap_value: value, which mask gap elements in array
    :param chain: TsForecastingChain object for filling in the gaps
    :param max_window_size: window length
    """

    def __init__(self, gap_value, chain, max_window_size: int = 50):
        super().__init__(gap_value)
        self.chain = chain
        self.max_window_size = max_window_size

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
                    TsForecastingParams(forecast_length=len_gap,
                                        max_window_size=self.max_window_size,
                                        return_all_steps=False,
                                        make_future_prediction=True))

        input_data = InputData(idx=np.arange(0, len(timeseries_train)),
                               features=None,
                               target=timeseries_train,
                               task=task,
                               data_type=DataTypesEnum.ts)

        available_model_types_primary = ['linear', 'ridge', 'lasso',
                                         'dtreg', 'knnreg']

        available_model_types_secondary = ['linear', 'ridge', 'lasso', 'rfr',
                                           'dtreg', 'knnreg', 'svr']

        composer_requirements = GPComposerRequirements(
            primary=available_model_types_primary,
            secondary=available_model_types_secondary, max_arity=5,
            max_depth=2, pop_size=10, num_of_generations=10,
            crossover_prob=0.8, mutation_prob=0.8,
            max_lead_time=datetime.timedelta(minutes=10),
            add_single_model_chains=True)

        metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.RMSE)
        builder = GPComposerBuilder(task=task).with_requirements(composer_requirements).with_metrics(metric_function).with_initial_chain(self.chain)
        composer = builder.build()

        obtained_chain = composer.compose_chain(data=input_data,
                                                is_visualise=False)

        # Making predictions for the missing part in the time series
        obtained_chain.__class__ = TsForecastingChain
        obtained_chain.fit_from_scratch(input_data)

        print('\nObtained chain')
        for node in obtained_chain.nodes:
            print(str(node))

        # "Test data" for making prediction for a specific length
        test_data = InputData(idx=np.arange(0, len_gap),
                              features=None,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.ts)

        predicted_values = obtained_chain.forecast(initial_data=input_data,
                                                   supplementary_data=test_data).predict
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

            # Gap-filling algorithm
            chain = TsForecastingChain()
            node_1_first = PrimaryNode('ridge')
            node_1_second = PrimaryNode('ridge')

            node_2_first = SecondaryNode('linear', nodes_from=[node_1_first])
            node_2_second = SecondaryNode('linear', nodes_from=[node_1_second])
            node_final = SecondaryNode('svr', nodes_from=[node_2_first,
                                                          node_2_second])
            chain.add_node(node_final)

            print('Init chain')
            for node in chain.nodes:
                print(str(node))

            gapfiller = ComposerGapFiller(gap_value=-100.0,
                                          chain=chain,
                                          max_window_size=50)
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
    main_dataframe.to_csv(file_with_results, index=False)



# Run the linear example
folder_to_save = 'D:/iccs_article/fedot_composer'
files_list = ['Synthetic.csv', 'Sea_hour.csv', 'Sea_10_240.csv']
columns_with_gap = ['gap', 'gap_center']
file_with_results = 'D:/iccs_article/fedot_composer_report.csv'

if __name__ == '__main__':
    run_fedot_composer(folder_to_save, files_list,
                       columns_with_gap, file_with_results, vis=True)