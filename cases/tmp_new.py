import sys
import traceback

from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder

from fedot.api.main import Fedot


def prepare_data(df_el, target_column):
    features, target = df_el.drop(target_column, axis=1).values, df_el[target_column].values

    return features, target, df_el.index.values


def call(target_column='species'):
    df_el, y = load_iris(return_X_y=True, as_frame=True)
    df_el[target_column] = LabelEncoder().fit_transform(y)

    features, target = df_el.drop(target_column, axis=1).values, df_el[target_column].values

    # p = np.random.permutation(len(features))
    # features, target = features[p], target[p]

    try:
        problem = 'classification'
        auto_model = Fedot(problem=problem, seed=42, timeout=2, composer_params={'metric': 'f1'})
        pipeline = auto_model.fit(features=features, target=target)
    except Exception as e:
        print(traceback.format_exc())
        raise e.with_traceback(sys.exc_info()[2])


if __name__ == '__main__':
    call()
