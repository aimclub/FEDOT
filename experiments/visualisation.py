import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def calculate_proportion(preprocessing_time: np.ndarray, full_time: np.ndarray):
    """ Calculate ratio of preprocessing """
    preprocessing_time = np.array(preprocessing_time)
    full_time = np.array(full_time)

    ratios = (preprocessing_time / full_time) * 100

    print(f'Mean ratio: {np.mean(ratios):.2f} %')


def display_plots(dir: str, name1: str, name2: str, title: str):
    """ Display some plots after experiments """
    dir = os.path.abspath(dir)
    df_1 = pd.read_csv(os.path.join(dir, f'{name1}.csv'))
    delim = 'using_cache'
    df_1[delim] = 1.
    df_2 = pd.read_csv(os.path.join(dir, f'{name2}.csv'))
    df_2[delim] = 0.
    df = pd.concat([df_1, df_2])
    df['Fit preprocessing ratio, %'] = (df['fit preprocessing'] / df['fit full time']) * 100
    df['Predict preprocessing ratio, %'] = (df['predict preprocessing'] / df['predict full time']) * 100

    for flag in [1., 0.]:
        cur_df = df[df[delim] == flag]
        mean_fit_preprocessing = np.mean(np.array(cur_df['Fit preprocessing ratio, %']))
        mean_predict_preprocessing = np.mean(np.array(cur_df['Predict preprocessing ratio, %']))
        print(
            f'Cache={bool(flag)},\n\t mean fitting time={mean_fit_preprocessing:.2f}, mean predicting time={mean_predict_preprocessing}')

    df[delim] = df[delim].astype(bool)
    sns.boxplot(x=delim, y="Fit preprocessing ratio, %", data=df,
                palette='Blues').set(title=title)
    # sns.boxplot(x=delim, y="Predict preprocessing ratio, %", data=df,
    #            palette='Blues').set(title=title)
    plt.show()


if __name__ == '__main__':
    ext = '_clean'
    if ext == '':
        title = '1 cat col + 20% of gaps, 20k dataset'
    elif ext == '_clean':
        title = 'no cats, no gaps, 20k dataset'
    display_plots('.', f'with_cache{ext}', f'without_cache{ext}', title)
