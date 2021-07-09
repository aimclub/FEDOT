from methods.gap_generator import generate_gaps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 18, 7


csv_file = 'data/cleaned_data.tsv'
columns = ['Security', 'Date', 'DateNum', 'Forward3m', 'Forward6m', 'Forward12m',
           'Premium3m', 'Premium6m', 'Premium12m', 'EBIT Growth_time',
           'Current Ratio_time', 'Market Cap_time', 'EV/EBITDA_time',
           'Div Yield_time', 'Interest Coverage_time', 'Rev Growth_time',
           'Analyst Rating_time', 'ROIC_time', 'Fin Lvg_time', 'P/Book_time',
           'P/FCF_time', 'EBIT Growth_all', 'Current Ratio_all', 'Market Cap_all',
           'EV/EBITDA_all', 'Div Yield_all', 'Interest Coverage_all',
           'Rev Growth_all', 'Analyst Rating_all', 'ROIC_all', 'Fin Lvg_all',
           'P/Book_all', 'P/FCF_all']

df = pd.read_csv(csv_file, sep=',')
columns = ['Forward3m', 'Forward6m', 'Forward12m',
           'Premium3m', 'Premium6m', 'Premium12m', 'EBIT Growth_time',
           'Current Ratio_time', 'Market Cap_time', 'EV/EBITDA_time',
           'Div Yield_time', 'Interest Coverage_time', 'Rev Growth_time',
           'Analyst Rating_time', 'ROIC_time', 'Fin Lvg_time', 'P/Book_time',
           'P/FCF_time', 'EBIT Growth_all', 'Current Ratio_all', 'Market Cap_all',
           'EV/EBITDA_all', 'Div Yield_all', 'Interest Coverage_all',
           'Rev Growth_all', 'Analyst Rating_all', 'ROIC_all', 'Fin Lvg_all',
           'P/Book_all', 'P/FCF_all']
for col in columns:
    plt.plot(df[col])
    plt.show()

# 30%
# generate_gaps(csv_file=csv_file,
#               gap_dict={550: 150,
#                         1000: 140,
#                         1600: 360,
#                         2500: 620,
#                         4050: 420,
#                         5400: 200},
#               gap_value=-100.0,
#               column_name='gap', sep='\t',
#               vis=True, column='EBIT Growth_time')
#
# generate_gaps(csv_file=csv_file,
#               gap_dict={2500: 1500},
#               gap_value=-100.0,
#               column_name='gap_center', sep='\t',
#               vis=True, column='DateNum')
