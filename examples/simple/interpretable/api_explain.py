import pandas as pd

from fedot.api.main import Fedot
from fedot.core.utils import fedot_project_root


def run_api_explain_example(visualization=False, timeout=None):
    train_data = pd.read_csv(f'{fedot_project_root()}/cases/data/cancer/cancer_train.csv', index_col=0)
    figure_path = 'api_explain_example.png'

    # Feature and class names for visualization
    feature_names = train_data.columns.tolist()
    target_name = feature_names.pop()
    target = train_data[target_name]
    class_names = target.unique().astype(str).tolist()

    # Building simple pipeline
    model = Fedot(problem='classification', timeout=timeout)
    model.fit(features=train_data, target=target_name, predefined_model='rf')

    # Current pipeline explaining
    explainer = model.explain(
        method='surrogate_dt', visualization=visualization,
        # The following parameters are only used if visualize == True:
        save_path=figure_path, dpi=200, feature_names=feature_names,
        class_names=class_names,
        precision=6
    )

    return explainer


if __name__ == '__main__':
    run_api_explain_example(visualization=True, timeout=5)
