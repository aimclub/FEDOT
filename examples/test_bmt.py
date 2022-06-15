import bamt.Preprocessors as pp
from bamt.Builders import StructureBuilder
import bamt.Networks as Nets
import pandas as pd
import sklearn.preprocessing as preprocessing
data = pd.read_csv('examples/data/'+'asia'+'.csv')
data.drop(['Unnamed: 0'], axis=1, inplace=True)
        # print(data.isna().sum())
data.dropna(inplace=True)
data.reset_index(inplace=True, drop=True)

encoder = preprocessing.LabelEncoder()
discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    # global p
p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
discretized_data, est = p.apply(data)


struct = [('asia', 'xray'), ('tub', 'bronc'), ('smoke', 'asia'), ('lung', 'either'), ('lung', 'xray'), ('bronc', 'dysp'), ('either', 'smoke'), ('xray', 'tub'), ('xray', 'dysp')]

bn = Nets.DiscreteBN()
bn.add_nodes(p.info)
bn.set_structure(edges=struct)
bn.fit_parameters(data)
for c in data.columns:
    print(c)
    test = data.drop(columns=[c])
    bn.predict(test)