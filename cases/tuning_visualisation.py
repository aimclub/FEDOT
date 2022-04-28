import os
from typing import Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from fedot.core.utils import project_root


def collect_datasets_results(dir, approaches):
    """ Parse folders and create dataframes for classification and regression tables

    :param dir: path to folder with saved results
    :param approaches: name of approaches
    """
    result_df_reg = []
    result_df_class = []
    for approach in approaches:
        folder = os.path.join(dir, approach)
        path_few_iterations = os.path.join(folder, '20')
        path_many_iterations = os.path.join(folder, '100')

        datasets = os.listdir(path_few_iterations)
        for dataset in datasets:
            df_20 = pd.read_csv(os.path.join(path_few_iterations, dataset))
            df_20['Iterations number'] = [20] * len(df_20)
            df_20['Approach'] = [approach.split('_')[0]] * len(df_20)

            df_100 = pd.read_csv(os.path.join(path_many_iterations, dataset))
            df_100['Iterations number'] = [100] * len(df_100)
            df_100['Approach'] = [approach.split('_')[0]] * len(df_100)

            if 'reg' in dataset:
                df_20['Dataset'] = [dataset[4: -4]] * len(df_20)
                df_100['Dataset'] = [dataset[4: -4]] * len(df_100)
                result_df_reg.extend([df_20, df_100])

            elif 'class' in dataset:
                df_20['Dataset'] = [dataset[6: -4]] * len(df_20)
                df_100['Dataset'] = [dataset[6: -4]] * len(df_100)
                result_df_class.extend([df_20, df_100])

    result_df_reg = pd.concat(result_df_reg)
    result_df_class = pd.concat(result_df_class)

    # Calculate improvements (deltas) - the bigger delta - the better
    result_df_reg['SMAPE improvement'] = result_df_reg['SMAPE before tuning'] - result_df_reg['SMAPE after tuning']
    result_df_class['ROC AUC improvement'] = result_df_class['ROC AUC after tuning'] - \
                                             result_df_class['ROC AUC before tuning']
    return result_df_reg, result_df_class


def kde_plot(iterations_number: Union[int, str], dataset_name: str, result_df: pd.DataFrame):
    """ Display KDE plot
    See https://seaborn.pydata.org/examples/kde_ridgeplot.html for more details
    """
    target_column = 'ROC AUC improvement'
    x_addition = 0.1
    if 'SMAPE improvement' in list(result_df.columns):
        target_column = 'SMAPE improvement'
        x_addition = 1.0

    iterations_number = int(iterations_number)
    dataset_results = result_df[result_df['Dataset'] == dataset_name]
    dataset_results = dataset_results[dataset_results['Iterations number'] == iterations_number]

    x_min = min(dataset_results[target_column])
    x_max = max(dataset_results[target_column])
    for approach in list(dataset_results['Approach'].unique()):
        print(approach)
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        approach_df = dataset_results[dataset_results['Approach'] == approach]
        # Initialize the FacetGrid object
        pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
        g = sns.FacetGrid(approach_df, row="Pipeline", hue="Pipeline", aspect=15, height=.5, palette=pal,
                          xlim=[x_min - x_addition, x_max + x_addition])

        # Draw the densities in a few steps
        g.map(sns.kdeplot, target_column, bw_adjust=.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
        g.map(sns.kdeplot, target_column, clip_on=False, color="w", lw=2, bw_adjust=.5)

        # passing color=None to refline() uses the hue mapping
        g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

        # Define and use a simple function to label the plot in axes coordinates
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, label, fontweight="bold", color=color,
                    ha="left", va="center", transform=ax.transAxes)

        g.map(label, target_column)

        # Set the subplots to overlap
        g.figure.subplots_adjust(hspace=-.25)

        # Remove axes details that don't play well with overlap
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)
        plt.show()


def display_mean_metrics(dataframe: pd.DataFrame, case: str):
    """ Calculate and display information about metrics """
    if case == 'regression':
        delta_column = 'SMAPE improvement'
        before_column = 'SMAPE before tuning'
    else:
        delta_column = 'ROC AUC improvement'
        before_column = 'ROC AUC before tuning'

    # Calculate improvements in percentages
    dataframe['Improvement, %'] = (dataframe[delta_column] / dataframe[before_column]) * 100
    for pipeline_type in list(dataframe['Pipeline'].unique()):
        pipeline_type_results = dataframe[dataframe['Pipeline'] == pipeline_type]

        for approach in list(pipeline_type_results['Approach'].unique()):
            approach_df = pipeline_type_results[pipeline_type_results['Approach'] == approach]

            mean_improvement = np.mean(np.array(approach_df['Improvement, %']))
            std_improvement = np.std(np.array(approach_df['Improvement, %']))

            print(f'Approach {approach} pipeline type {pipeline_type}: '
                  f'mean improvement {mean_improvement:.1f} % ± std {std_improvement:.1f}')
        print('\n')


def generate_sum_column(dataframe):
    case_column = []
    for i in range(len(dataframe)):
        row = dataframe.iloc[i]
        new_value = ''.join(('Pipeline: ', str(row['Pipeline']),
                             '; dataset: ', str(row['Dataset'])))
        case_column.append(new_value)
    dataframe['Case'] = case_column

    return dataframe


def perform_visual_analysis(working_dir: str):
    working_dir = os.path.abspath(working_dir)
    folders = os.listdir(working_dir)
    tuner_folders = list(filter(lambda x: 'tuner' in x, folders))

    print(f'Calculated tuning variations: {tuner_folders}')

    result_df_reg, result_df_class = collect_datasets_results(dir=working_dir, approaches=tuner_folders)
    print('\n--- Regression case ---')
    filtered_df = result_df_reg[result_df_reg['Dataset'] != 'pol'].copy()
    display_mean_metrics(filtered_df, 'regression')

    print('\n--- Classification case ---')
    display_mean_metrics(result_df_class, 'classification')

    kde_plot(iterations_number=20, dataset_name='Amazon_employee_access', result_df=result_df_class)

    with sns.axes_style("darkgrid"):
        sns.catplot(x='Approach', y='SMAPE improvement',
                    hue='Pipeline', col='Iterations number',
                    row='Dataset',
                    palette='Spectral',
                    data=result_df_reg,
                    kind="strip", dodge=True,
                    height=4, aspect=.7)
        plt.show()

        sns.catplot(x='Approach', y='ROC AUC improvement',
                    hue='Pipeline', col='Iterations number',
                    row='Dataset',
                    palette='Paired',
                    data=result_df_class,
                    kind="strip", dodge=True,
                    height=4, aspect=.7)
        plt.show()

        df_for_vis = generate_sum_column(result_df_reg)
        # Filter dataframe - remain only representative cases
        df_for_vis = df_for_vis[df_for_vis['Pipeline'] != 'C']
        df_for_vis = df_for_vis[df_for_vis['Dataset'] != 'pol']

        sns.catplot(x='Approach', y='Time, sec.',
                    hue='Iterations number',
                    col='Case',
                    palette='rainbow',
                    data=df_for_vis,
                    kind="violin", dodge=True,
                    split=False,
                    height=4, aspect=.9)
        plt.show()


if __name__ == '__main__':
    perform_visual_analysis(working_dir=os.path.join(project_root(), 'tuning'))
