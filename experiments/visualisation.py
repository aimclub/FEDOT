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
    preprocessing_df = pd.read_csv(os.path.join(dir, 'preprocessed_results.csv'))
    preprocessing_df['was preprocessed'] = [True] * len(preprocessing_df)
    preprocessing_df['Fit preprocessing ratio, sec.'] = (preprocessing_df['fit preprocessing'] / preprocessing_df['fit full time']) * 100
    preprocessing_df['Predict preprocessing ratio, sec.'] = (preprocessing_df['predict preprocessing'] / preprocessing_df['predict full time']) * 100

    print(f'Preprocessing was set as True')
    calculate_proportion(preprocessing_df['fit preprocessing'], preprocessing_df['fit full time'])
    calculate_proportion(preprocessing_df['predict preprocessing'], preprocessing_df['predict full time'])

    non_preprocessing_df = pd.read_csv(os.path.join(dir, 'non_preprocessed_results.csv'))
    non_preprocessing_df['was preprocessed'] = [False] * len(non_preprocessing_df)
    non_preprocessing_df['Fit preprocessing ratio, sec.'] = (non_preprocessing_df['fit preprocessing'] / non_preprocessing_df['fit full time']) * 100
    non_preprocessing_df['Predict preprocessing ratio, sec.'] = (non_preprocessing_df['predict preprocessing'] / non_preprocessing_df['predict full time']) * 100

    print(f'\nPreprocessing was set as False')
    calculate_proportion(non_preprocessing_df['fit preprocessing'], non_preprocessing_df['fit full time'])
    calculate_proportion(non_preprocessing_df['predict preprocessing'], non_preprocessing_df['predict full time'])

    common_df = pd.concat([preprocessing_df, non_preprocessing_df])

    sns.boxplot(x="was preprocessed", y="Fit preprocessing ratio, sec.", data=common_df,
                palette='Blues')
    plt.show()

    sns.boxplot(x="was preprocessed", y="Predict preprocessing ratio, sec.", data=common_df,
                palette='Reds')
    plt.show()


if __name__ == '__main__':
    display_plots('.')
