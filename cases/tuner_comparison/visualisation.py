import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def get_complete_dataset(dir_name):
    results_df = []
    tuners = ['IOptTuner', 'SimultaneousTuner']
    iter_nums = [20, 100]
    for tuner in tuners:
        for iter_num in iter_nums:
            current_dir = os.path.join(dir_name, f'{tuner}_{iter_num}')
            datasets = os.listdir(current_dir)
            for dataset in datasets:
                df = pd.read_csv(os.path.join(current_dir, dataset))
                df['tuner'] = [tuner] * len(df)
                results_df.append(df)
    final_df = pd.concat(results_df, ignore_index=True, axis=0)
    final_df = final_df.replace({'class_Amazon_employee_access.csv': 'Amazon_employee_access',
                                 'class_cnae-9.csv': 'class_cnae-9'})
    is_regression = dir_name == 'regression'
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


def get_metric_statistics(results_df):
    iopt_mean = (results_df[results_df.tuner == 'IOptTuner']
                 .groupby(['dataset', 'iter_num', 'pipeline_type'])[['metric_improvement']].mean()
                 .rename(columns={'metric_improvement': 'IOpt mean'}))
    iopt_std = (results_df[results_df.tuner == 'IOptTuner']
                .groupby(['dataset', 'iter_num', 'pipeline_type'])[['metric_improvement']].std()
                .rename(columns={'metric_improvement': 'IOpt std'}))

    hopt_mean = (results_df[results_df.tuner == 'SimultaneousTuner']
                 .groupby(['dataset', 'iter_num', 'pipeline_type'])[['metric_improvement']].mean()
                 .rename(columns={'metric_improvement': 'Hyperopt mean'}))
    hopt_std = (results_df[results_df.tuner == 'SimultaneousTuner']
                .groupby(['dataset', 'iter_num', 'pipeline_type'])[['metric_improvement']].std()
                .rename(columns={'metric_improvement': 'Hyperopt std'}))

    stats_df = pd.concat([iopt_mean, hopt_mean, iopt_std, hopt_std], axis=1)
    pd.set_option('display.max_columns', None)
    return stats_df


if __name__ == '__main__':
    df = get_complete_dataset('regression')
    print(get_metric_statistics(df))
    plot_metric_improvements(df)
    plot_tuning_time(df)
