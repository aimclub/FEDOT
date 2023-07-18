import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def get_complete_dataset(dir_name, mean_time: bool = False, tuners_names=None):
    tuners = tuners_names or ['IOptTuner', 'SimultaneousTuner']
    results_df = []
    iter_nums = [20, 100]
    for tuner in tuners:
        for iter_num in iter_nums:
            current_dir = os.path.join(dir_name, f'{tuner}_{iter_num}')
            datasets = os.listdir(current_dir)
            for dataset in datasets:
                file_name = dataset
                if mean_time and dataset.startswith('mean_time'):
                    print(dataset)
                    df = pd.read_csv(os.path.join(current_dir, file_name))
                    df['iter_num'] = [iter_num] * len(df)
                elif not dataset.startswith('mean_time'):
                    df = pd.read_csv(os.path.join(current_dir, file_name))
                df['tuner'] = [tuner] * len(df)
                results_df.append(df)
    final_df = pd.concat(results_df, ignore_index=True, axis=0)
    final_df = final_df.replace({'class_Amazon_employee_access.csv': 'Amazon_employee_access',
                                 'class_cnae-9.csv': 'class_cnae-9'})
    is_regression = dir_name == 'regression' or 'forecasting'
    final_df['metric_improvement'] = final_df['final_metric'] - final_df['init_metric']
    if is_regression:
        final_df['metric_improvement'] = -final_df['metric_improvement']

    return final_df


def plot_metric_improvements(results_df):
    with sns.axes_style("darkgrid"):
        sns.catplot(x='tuner', y='metric_improvement',
                    hue='pipeline_type', col='iter_num',
                    row='dataset',
                    palette='Spectral',
                    data=results_df,
                    kind="strip", dodge=True,
                    height=3.5, aspect=1.7)
        plt.show()


def plot_tuning_time(results_df):
    results_df['Case'] = results_df.apply(
        lambda x: f'Pipeline: {str(x["pipeline_type"])}; dataset: {str(x["dataset"])}',
        axis=1)
    with sns.axes_style("darkgrid"):
        sns.catplot(y='tuning_time', x='tuner',
                    hue='iter_num',
                    row='dataset',
                    col='pipeline_type',
                    palette='rainbow',
                    data=results_df,
                    kind="violin", dodge=True,
                    split=False,
                    height=3.5, aspect=1.5)
        plt.show()


def get_metric_statistics(results_df, task, tuners_names=None):
    tuners_names = tuners_names or ['IOptTuner', 'SimultaneousTuner']
    means = []
    stds = []
    for tuner in tuners_names:
        mean = (results_df[results_df.tuner == tuner]
                .groupby(['dataset', 'iter_num', 'pipeline_type'])[['metric_improvement']].mean()
                .rename(columns={'metric_improvement': f'{tuner} mean'}))
        means.append(mean)
        std = (results_df[results_df.tuner == tuner]
               .groupby(['dataset', 'iter_num', 'pipeline_type'])[['metric_improvement']].std()
               .rename(columns={'metric_improvement': f'{tuner} std'}))
        stds.append(std)

    stats_df = pd.concat([*means, *stds], axis=1)
    pd.set_option('display.max_columns', None)
    stats_df.to_csv(os.path.join(task, 'statistics.csv'))
    return stats_df


def get_mean_time(results_df, task, tuners_names=None):
    tuners_names = tuners_names or ['IOptTuner', 'SimultaneousTuner']
    time_means = []
    for tuner in tuners_names:
        mean = (results_df[results_df.tuner == tuner]
                .groupby(['dataset', 'iter_num', 'pipeline_type'])[['tuning_time']].mean()
                .rename(columns={'tuning_time': f'{tuner} time mean'}))
        time_means.append(mean)
    stats_df = pd.concat(time_means, axis=1)
    pd.set_option('display.max_columns', None)
    stats_df.to_csv(os.path.join(task, 'mean_time.csv'))
    return stats_df


if __name__ == '__main__':
    task = 'forecasting'
    tuners = ['IOptTuner', 'SimultaneousTuner', 'OptunaTuner']
    df = get_complete_dataset(task, mean_time=False, tuners_names=tuners)
    print(get_metric_statistics(df, task, tuners))
    print(get_mean_time(df, task, tuners))
    plot_metric_improvements(df)
    plot_tuning_time(df)
