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
    return final_df


def plot_metric_improvements(results_df):
    df['Metric improvement'] = df['final_metric'] - df['init_metric']
    with sns.axes_style("darkgrid"):
        sns.catplot(x='tuner', y='Metric improvement',
                    hue='pipeline_type', col='iter_num',
                    row='dataset',
                    palette='Spectral',
                    data=results_df,
                    kind="strip", dodge=True,
                    height=3.5, aspect=1.7)
        plt.show()


def plot_tuning_time(results_df):
    results_df['Case'] = results_df.apply(lambda x: f'Pipeline: {str(x["pipeline_type"])}; dataset: {str(x["dataset"])}',
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


if __name__ == '__main__':
    df = get_complete_dataset('')
    plot_metric_improvements(df)
    plot_tuning_time(df)
