
from copy import deepcopy
import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
from pyitlib import discrete_random_variable as drv
from requests import delete
from sklearn import preprocessing
import seaborn as sns
import itertools

import bamt.Preprocessors as pp

data_cancer = pd.read_csv(r'examples/data/cancer.csv')
data_cancer.drop(['Unnamed: 0'], axis=1, inplace=True)
vertices_cancer = list(data_cancer.columns)


encoder = preprocessing.LabelEncoder()
discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
discretized_data, est = p.apply(data_cancer)

# не воспринимает int64 как int, только int32
for i in discretized_data.columns:
    if discretized_data[i].dtype.name == 'int64':
        discretized_data = discretized_data.astype({i:'int32'})  

# data_earthquake = pd.read_csv(r'examples/data/earthquake.csv')
# data_earthquake.drop(['Unnamed: 0'], axis=1, inplace=True)
# vertices_earthquake = list(data_earthquake.columns)
# print(discretized_data)

empty = []
true_str = [('Pollution', 'Cancer'), ('Smoker', 'Cancer'), ('Cancer', 'Xray'), ('Cancer', 'Dyspnoea')]



def mi(stru, vertices):
    MI = 0
    for i in vertices:
        if i in list(itertools.chain(*stru)):
            arcs = [j for j in stru if j[0]==i or j[1]==i]
            for a in arcs:
                MI += drv.information_mutual(discretized_data[a[0]].values, discretized_data[a[1]].values)
        else: 
            MI += drv.information_mutual(discretized_data[i].values, discretized_data[i].values)
    return(MI)
    
print(mi(true_str, vertices_cancer))
print(mi(empty, vertices_cancer))


def cmi(stru, vertices):
    CMI = 0
    for X in vertices:
        if X in [k[1] for k in stru]:
            parents = [j[0] for j in stru if j[1]==X]
            if len(parents)==1:
                Y=parents[0]
                CMI += drv.information_mutual(discretized_data[X].values, discretized_data[Y].values)
            else:
                for Y in parents:
                    par = deepcopy(parents)
                    par.remove(Y)
                    df_Z_int=deepcopy(discretized_data[par])
                    df_Z_str = df_Z_int.astype(str)
                    df_Z = df_Z_str.agg('-'.join, axis=1)
                    enc = preprocessing.LabelEncoder()
                    Z = enc.fit_transform(df_Z)
                    df = deepcopy(discretized_data[[X,Y]])
                    df.insert(2, 'Z', Z)
                    CMI += drv.information_mutual_conditional(df[X].values, df[Y].values, df['Z'].values)
        else: 
            CMI += drv.information_mutual(discretized_data[X].values, discretized_data[X].values)
    return(CMI) 

print(cmi(true_str, vertices_cancer))
print(cmi(empty, vertices_cancer))
# X='Smoker'
# parents = ['Cancer', 'Xray', 'Dyspnoea']
# Y='Cancer'
# parents.remove(Y)
# df_Z_int=deepcopy(discretized_data[parents])
# df_Z_str = df_Z_int.astype(str)
# df_Z = df_Z_str.agg('-'.join, axis=1)
# enc = preprocessing.LabelEncoder()
# Z = enc.fit_transform(df_Z)
# df = deepcopy(discretized_data[[X,Y]])
# df.insert(2, 'Z', Z)
["Asia","Smoking","Tuberculosis","LungCancer","Bronchitis","Either","ChestXRay","Dyspnoea"

]
# Y1 = discretized_data.iloc[:,0].values

# print(drv.information_mutual(Y1,Y1))

# Y2 = discretized_data.iloc[:,1].values

# print(drv.information_mutual(Y1,Y2))
# print(drv.information_mutual(Y2,Y1))



# Y2 = discretized_data.iloc[:,3].values
# print(Y2)
# print(drv.information_mutual(Y2,Y2))
# print(discretized_data)

# print(drv.information_mutual(Y1,Y2))
# print([type(i) for i in discretized_data.iloc[:,0].values])

# print(drv.information_mutual(discretized_data.iloc[:,1].values, discretized_data.iloc[:,1].values))
# print(drv.information_mutual(discretized_data[vertices_cancer[0]].values, discretized_data[vertices_cancer[1]].values))


# print(drv.information_mutual(data_cancer[vertices_cancer[0]].values, list(map(lambda x: str(x), data_cancer[vertices_cancer[1]].values))))


# print(drv.information_mutual(data_cancer[vertices_cancer[1]].values, data_cancer[vertices_cancer[2]].values))
# print(drv.information_mutual(data_cancer[vertices_cancer[0]].values, data_cancer[vertices_cancer[1]].values))