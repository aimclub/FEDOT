import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_proportion(preprocessing_time: np.ndarray, full_time: np.ndarray):
    """ Calculate ratio of preprocessing """
    preprocessing_time = np.array(preprocessing_time)
    full_time = np.array(full_time)

    ratios = (preprocessing_time / full_time) * 100

    print(f'Mean ratio: {np.mean(ratios):.2f} %')


def display_plots(dir: str):
    """ Display some plots after experiments """
    dir = os.path.abspath(dir)
    df_1 = pd.read_csv(os.path.join(dir, 'default_preprocessors.csv'))
    df_2 = pd.read_csv(os.path.join(dir, 'no_default_preprocessors.csv'))
    df = pd.concat([df_1, df_2])
    df['Fit preprocessing ratio, %'] = (df['fit preprocessing'] / df['fit full time']) * 100
    df['pipeline structure'][df['pipeline structure'] != 'no preprocessing operations'] = 'preprocessing in the pipeline'

    df_1 = df[df['pipeline structure'] == 'no preprocessing operations']
    mean_preprocessing_default = np.mean(np.array(df_1['Fit preprocessing ratio, %']))
    print(f'Default preprocessing {mean_preprocessing_default:.2f}')

    df_2 = df[df['pipeline structure'] == 'preprocessing in the pipeline']
    mean_preprocessing_in_pipeline = np.mean(np.array(df_2['Fit preprocessing ratio, %']))
    print(f'Preprocessing in the pipeline {mean_preprocessing_in_pipeline:.2f}')

    sns.boxplot(x="pipeline structure", y="Fit preprocessing ratio, %", data=df,
                palette='Blues')
    plt.show()


if __name__ == '__main__':
    display_plots('.')
