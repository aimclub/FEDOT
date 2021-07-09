from methods.gap_generator import generate_gaps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 18, 7


csv_file = 'data/oil.csv'
# 30%
generate_gaps(csv_file=csv_file,
              gap_dict={100: 20, 250: 50, 430: 25},
              gap_value=-100.0,
              column_name='gap', column='Height',
              vis=True)

generate_gaps(csv_file=csv_file,
              gap_dict={400: 150},
              gap_value=-100.0,
              column_name='gap_center', column='Height',
              vis=True)

