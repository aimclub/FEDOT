import os

import pandas as pd

from examples.simple.classification.classification_pipelines import classification_complex_pipeline
from fedot.core.data.data import InputData
from fedot.core.utils import fedot_project_root
from fedot.explainability.explainers import explain_pipeline


def run_pipeline_explain():
    # Specifying paths
    train_data_path = os.path.join(fedot_project_root(), 'cases', 'data', 'cancer', 'cancer_train.csv')
    figure_path = 'pipeline_explain_example.png'

    # Feature and class names for visualization
    feature_names = pd.read_csv(train_data_path, index_col=0, nrows=0).columns.tolist()
    target_name = feature_names.pop()
    target = pd.read_csv(train_data_path, usecols=[target_name])[target_name]
    class_names = target.unique().astype(str).tolist()

    # Data load
    train_data = InputData.from_csv(train_data_path)

    # Pipeline composition
    pipeline = classification_complex_pipeline()

    # Pipeline fitting
    pipeline.fit(train_data)

    # Pipeline explaining
    explainer = explain_pipeline(pipeline, data=train_data, method='surrogate_dt', visualization=True)

    # Visualizing explanation and saving the plot
    print(f'Built surrogate model: {explainer.surrogate_str}')
    explainer.visualize(save_path=figure_path, dpi=200, feature_names=feature_names, class_names=class_names,
                        precision=6)


if __name__ == '__main__':
    run_pipeline_explain()
