import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer


def scaling_preprocess(x):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(x)
    x = imp.transform(x)
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(x)

def normalize_preprocess(x):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(x)
    x = imp.transform(x)
    return preprocessing.normalize(x)

def simple_preprocess(x):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(x)
    x = imp.transform(x)
    return x
