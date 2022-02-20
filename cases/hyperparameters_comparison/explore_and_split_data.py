import pandas as pd
import os

from fedot.core.utils import fedot_project_root
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

test_size = 0.2

overview = pd.DataFrame(columns=['name', 'rows', 'num_features', 'cat_features', 'num_classes'])

data_sources = pd.read_csv('data_for_comparison/data_sources.csv', index_col='name', low_memory=True)

short_path = 'data_for_comparison/original_datasets'
for filename in os.listdir(short_path):
    data = pd.read_csv(short_path + '/' + filename)
    data.rename(columns={data.columns.values[-1]: 'target'}, inplace=True)
    if 'ID' in data.columns:
        data.drop(columns='ID', inplace=True)
    if 'id' in data.columns:
        data.drop(columns='id', inplace=True)
    if 'index' in data.columns:
        data.drop(columns='index', inplace=True)
    if 'Index' in data.columns:
        data.drop(columns='Index', inplace=True)

    name = filename.replace('.csv', '')
    task = data_sources.loc[name]['task']
    rows = data.shape[0]

    features_dtypes = data.iloc[:, :-1].dtypes.astype(str).value_counts().to_dict()

    cat_features = features_dtypes['object'] if 'object' in features_dtypes else 0
    num_features = data.shape[1] - cat_features - 1
    if task == 'classification':
        data['target'] = OrdinalEncoder().fit_transform(data[['target']])
        num_classes = data['target'].nunique()
    else:
        num_classes = -1

    overview = overview.append({'name': name, 'rows': rows,
                                'num_features': num_features,
                                'cat_features': cat_features,
                                'num_classes': num_classes},
                               ignore_index=True)

    train, test = train_test_split(data,
                                   test_size=test_size,
                                   random_state=16777216,
                                   stratify=data['target'] if task == 'classification' else None)

    train.index = [i for i in range(train.shape[0])]
    test.index = [i for i in range(test.shape[0])]
    train.reset_index(inplace=True)
    test.reset_index(inplace=True)
    train.columns = ['ID'] + [c for c in train.columns][1:]
    test.columns = ['ID'] + [c for c in test.columns][1:]
    train.set_index('ID', inplace=True)
    test.set_index('ID', inplace=True)

    train.to_csv('data_for_comparison/train_datasets/' + filename)
    test.to_csv('data_for_comparison/test_datasets/' + filename)

overview = overview.merge(data_sources.reset_index())
overview.to_csv('data_for_comparison/overview.csv', index=False)
