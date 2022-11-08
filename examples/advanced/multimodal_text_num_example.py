from pathlib import Path

from fedot.api.main import Fedot
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.utils import fedot_project_root


def run_multi_modal_example(file_path: str, visualization=False) -> float:
    """
    This is an example of FEDOT use on multimodal data.
    The data is taken and adapted from Wine Reviews dataset (winemag-data_first150k):
    https://www.kaggle.com/datasets/zynicide/wine-reviews
    and contains information about wine country, region, price, etc.
    Column that contains text features is 'description'.
    Other columns contain numerical and categorical features.
    The aim is to predict wine variety, so it's a classification task.

    :param file_path: path to the file with multimodal data
    :param visualization: if True, then final pipeline will be visualised

    :return: F1 metrics of the model
    """
    task = 'classification'
    path = Path(fedot_project_root(), file_path)
    data = MultiModalData.from_csv(file_path=path, task=task, target_columns='variety', index_col=None)
    fit_data, predict_data = train_test_data_setup(data, shuffle_flag=True, split_ratio=0.7)

    automl_model = Fedot(problem=task, timeout=10)
    automl_model.fit(features=fit_data,
                     target=fit_data.target)

    prediction = automl_model.predict(predict_data)
    metrics = automl_model.get_metrics()

    if visualization:
        automl_model.current_pipeline.show()

    print(f'F1 for validation sample is {round(metrics["f1"], 3)}')

    return metrics["f1"]


if __name__ == '__main__':
    run_multi_modal_example(file_path='examples/data/multimodal_wine.csv', visualization=True)
