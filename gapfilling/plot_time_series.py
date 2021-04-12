import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 18, 7

import warnings
warnings.filterwarnings('ignore')

csv_file = 'data/Traffic.csv'
dataframe = pd.read_csv(csv_file)
dataframe['Date'] = pd.to_datetime(dataframe['Date'])
plt.plot(dataframe['Date'], dataframe['Height'])
plt.xlabel('Date', fontsize=13)
plt.ylabel('Traffic volume', fontsize=13)
plt.show()

csv_file = 'data/Temperature.csv'
dataframe = pd.read_csv(csv_file)
dataframe['Date'] = pd.to_datetime(dataframe['Date'])
plt.plot(dataframe['Date'], dataframe['Height'])
plt.xlabel('Date', fontsize=13)
plt.ylabel('Temperature', fontsize=13)
plt.show()


csv_file = 'data/Sea_hour.csv'
dataframe = pd.read_csv(csv_file)
dataframe['Date'] = pd.to_datetime(dataframe['Date'])
plt.plot(dataframe['Date'], dataframe['Height'])
plt.xlabel('Date', fontsize=13)
plt.ylabel('Sea surface height, m', fontsize=13)
plt.show()


csv_file = 'data/Sea_10_240.csv'
dataframe = pd.read_csv(csv_file)
dataframe['Date'] = pd.to_datetime(dataframe['Date'])
plt.plot(dataframe['Date'], dataframe['Height'])
plt.xlabel('Date', fontsize=13)
plt.ylabel('Sea surface height, m', fontsize=13)
plt.show()
