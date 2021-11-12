import pandas as pd
import numpy as np

from cases.medical.wrappers import display_metrics

path_to_file = 'covid_forecasts.csv'
df = pd.read_csv(path_to_file)
display_metrics(np.array(df['forecast']), np.array(df['actual']))
