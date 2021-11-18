import pandas as pd
import numpy as np

from cases.medical.wrappers import display_metrics


def calculate_metrics_from_file(path_to_file: str):
    df = pd.read_csv(path_to_file)
    display_metrics(np.array(df['forecast']), np.array(df['actual']))


calculate_metrics_from_file(path_to_file='flu_forecasts.csv')
