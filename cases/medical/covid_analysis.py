import os
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error
from cases.medical.wrappers import smape
from matplotlib import pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 6, 5


TEST_RATIO = 0.3


class CovidAnalyser:
    """ Class for analysing time series forecasts for COVID data """
    metric_by_name = {'mae': mean_absolute_error,
                      'smape': smape}

    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        self.regions_info = {}
        self.regions_paths = {}

    def find_best_pipelines(self, metric: str):
        """ Find best pipelines by desired metric """
        metric_func = self.metric_by_name.get(metric)

        regions = os.listdir(self.work_dir)

        for region in regions:
            print('\n------------------------------------------------------------')
            print(f'Region {region}')
            print('------------------------------------------------------------')
            region_path = os.path.join(self.work_dir, region)

            train_fractions = os.listdir(region_path)
            for train_fraction in train_fractions:
                ratio = train_fraction.replace('_', '.')
                ratio_path = os.path.join(region_path, train_fraction)

                all_files = os.listdir(ratio_path)
                ratio_info = []
                for file in all_files:
                    if file.endswith('.csv'):
                        # It is csv file with forecast
                        df = pd.read_csv(os.path.join(ratio_path, file))
                        # Calculate metric value
                        metric = metric_func(np.array(df['actual']), np.array(df['forecast']))

                        splitted_name = file.split('_')
                        last_part = splitted_name[-1].split('.')
                        launch_id = int(last_part[0])

                        ratio_info.append([launch_id, metric])

                ratio_info = pd.DataFrame(ratio_info, columns=['Launch', 'Metric'])
                ratio_info = ratio_info.sort_values(by=['Metric'])

                print(f'\nRegion {region} Ratio {ratio}')
                print(ratio_info.head(2))

                # Update info
                region_ratio_id = ''.join((str(region), '|', str(ratio)))
                self.regions_info.update({region_ratio_id: ratio_info})
                self.regions_paths.update({region_ratio_id: ratio_path})

    def plot_best_forecast(self, region, ratio, launch_id=None):
        """ Display forecast plot for desired train ratio and region """
        if launch_id is None:
            # Automatically detect best try
            key = ''.join((str(region), '|', str(ratio)))
            ratio_info = self.regions_info[key]
            launch_id = round(ratio_info.iloc[0]['Launch'])
            print(f'Best try for ratio {ratio} was during launch {launch_id}')

        if '_' in region:
            # Region contains not only country but city also
            splitted_region = region.split('_')
            df_source = self._load_source_dataframe(splitted_region[0], splitted_region[-1])
            title = f'\nRegion {splitted_region[0]} Province {splitted_region[-1]}'
        else:
            df_source = self._load_source_dataframe(region)
            title = f'\nRegion {region}'

        ratio_path = os.path.join(self.work_dir, region, str(ratio).replace('.', '_'))
        files = os.listdir(ratio_path)
        for file in files:
            if file.endswith('.csv') and str(launch_id) in file:
                # Good file
                df = pd.read_csv(os.path.join(ratio_path, file), parse_dates=['datetime'])

                full_used_ts_size = float(ratio) + TEST_RATIO
                full_used_ts_len = round(len(df_source['actual']) * full_used_ts_size)
                test_len = round(len(df_source['actual']) * TEST_RATIO)

                print(f'Train ts length: {full_used_ts_len - test_len} (elements)')
                plt.plot(df_source['datetime'], df_source['actual'], label='Actual time series')
                plt.plot(df['datetime'], df['forecast'], label=f'Forecast for 30 days ahead')
                plt.plot(df_source.iloc[-full_used_ts_len:-test_len]['datetime'],
                         df_source.iloc[-full_used_ts_len:-test_len]['actual'],
                         label='Part used for train', alpha=0.8, linewidth=4)
                plt.title(f'In-sample validation. Covid case. {title}')
                plt.xlabel('Datetime')
                plt.legend()
                plt.grid()
                plt.show()

    def metrics_for_best_forecast(self):
        raise NotImplementedError()

    def display_mean_metrics(self, region=None):
        """ Display MAE, RMSE and SMAPE in tabular form for considering region """

        final_table = self._generate_table_with_metrics(average=True)
        if region is not None:
            # Show results for particular region
            final_table = final_table[final_table['region'] == region]
            print(final_table)
        else:
            # Calculate mean SMAPE metric
            sizes = final_table['train_size'].unique()
            for train_size in sizes:
                local_df = final_table[final_table['train_size'] == train_size]
                smapes = np.array(local_df['smape'], dtype=float)
                mean_smape = np.mean(smapes)

                print(f'Size {train_size}: SMAPE {mean_smape:.2f}')

    def plot_metric_vs_train_size(self):
        metrics_df = self._generate_table_with_metrics(average=False)
        metrics_df = metrics_df.astype({'smape': 'float64', 'train_size': 'float64'})

        # Remove outliers
        metrics_df = metrics_df[metrics_df['smape'] < 100]

        with sns.axes_style("darkgrid"):
            sns.catplot(x='train_size', y='smape',
                        hue='region', col='region',
                        data=metrics_df, kind="strip", dodge=False,
                        height=4, aspect=.7)
            plt.show()

    @staticmethod
    def _load_source_dataframe(country, city=None):
        df = pd.read_csv('time_series_covid19_confirmed_global.csv')
        df_country = df[df['Country/Region'] == country]
        if city is not None:
            df_country = df_country[df_country['Province/State'] == city]

        first_row = np.array(df_country.iloc[0])
        dates_df = pd.DataFrame({'datetime': np.array(df.columns[4:], dtype=str),
                                 'actual': np.array(first_row[4:], dtype=int)})
        dates_df['datetime'] = pd.to_datetime(dates_df['datetime'], format="%m/%d/%y")
        return dates_df

    def _generate_table_with_metrics(self, average: bool = True):
        i = 0
        for region_ratio, folder_path in self.regions_paths.items():
            current_region, ratio = region_ratio.split('|')

            all_files = os.listdir(folder_path)
            metrics = []
            for file in all_files:
                if file.endswith('.csv'):
                    # It is csv file with forecast
                    df = pd.read_csv(os.path.join(folder_path, file))
                    actual = np.array(df['actual'])
                    forecast = np.array(df['forecast'])

                    mae_value = mean_absolute_error(actual, forecast)
                    rmse_value = mean_squared_error(actual, forecast, squared=False)
                    smape_value = smape(actual, forecast)
                    metrics.append([mae_value, rmse_value, smape_value])

            metrics = np.array(metrics)
            if average is True:
                # Average results pf algorithm
                metrics = list(np.round(metrics.mean(axis=0), 2))
                metrics.append(ratio)
                metrics.append(current_region)

            else:
                # Return all values
                ratios = np.array([ratio] * len(metrics)).reshape(-1, 1)
                regions = np.array([current_region] * len(metrics)).reshape(-1, 1)
                metrics = np.hstack((metrics, ratios, regions))

            if i == 0:
                final_table = metrics
                i += 1
            else:
                final_table = np.vstack((final_table, metrics))
                i += 1

        final_table = pd.DataFrame(final_table, columns=['mae', 'rmse', 'smape', 'train_size', 'region'])

        return final_table


if __name__ == '__main__':
    analyser = CovidAnalyser('./covid')
    analyser.find_best_pipelines(metric='smape')

    # Display best model forecast for wanted region and ratio
    # analyser.plot_best_forecast(region='Russia', ratio=0.7, launch_id=None)

    # Show metrics for desired region
    analyser.display_mean_metrics(region=None)

    # Make visualisations
    # analyser.plot_metric_vs_train_size()
