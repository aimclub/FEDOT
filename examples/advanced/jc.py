import os
from pathlib import Path
# from sklearnex import patch_sklearn
# patch_sklearn()
from fedot.api.main import Fedot
import numpy as np
import pandas as pd
from examples.simple.pipeline_import_export import create_correct_path
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import TaskTypesEnum, Task
from fedot.core.utils import fedot_project_root


def load_pipeline(path: str):
    # 0.
    # path = create_correct_path(path)
    pipeline = Pipeline().load(path)
    return pipeline


def run_multi_modal_example(file_path: str, is_visualise=True) -> float:
    """
    This is an example of FEDOT use on multimodal data.
    The data is taken and adapted from Wine Reviews dataset (winemag-data_first150k):
    https://www.kaggle.com/datasets/zynicide/wine-reviews
    and contains information about wine country, region, price, etc.
    Column that contains text features is 'description'.
    Other columns contain numerical and categorical features.
    The aim is to predict wine variety, so it's a classification task.

    :param file_path: path to the file with multimodal data
    :param is_visualise: if True, then final pipeline will be visualised

    :return: F1 metrics of the model
    """
    task = 'regression'
    train_path = Path(fedot_project_root(), file_path, 'jc_penney_products_train.csv')
    test_path = Path(fedot_project_root(), file_path, 'jc_penney_products_test.csv')
    fit_data = MultiModalData.from_csv(file_path=train_path, task=task, index_col=None,
                                       text_columns=['name_title', 'description', 'brand'],
                                       target_columns='sale_price')
    predict_data = MultiModalData.from_csv(file_path=test_path, task=task, index_col=None,
                                           text_columns=['name_title', 'description', 'brand'],
                                           target_columns='sale_price')

    path = 'examples/advanced/October-19-2022,02-13-11,AM pipeline_jc'
    loaded_pipeline = load_pipeline(os.path.join(fedot_project_root(), path))
    loaded_pipeline.fit(fit_data)
    # predict = loaded_pipeline.predict(predict_data).predict
    automl_model = Fedot(problem='regression',
                         timeout=15,
                         n_jobs=6,
                         safe_mode=False,
                         metric='r2',
                         preset='best_quality',
                         with_tuning=False)

    automl_model.fit(features=fit_data,
                     target=fit_data.target,
                     predefined_model=loaded_pipeline)  # 0.575

    prediction = automl_model.predict(predict_data)
    r2 = r2_score(predict_data.target, prediction)
    metrics = automl_model.get_metrics()
    if is_visualise:
        automl_model.current_pipeline.show(engine='pyvis')
    # automl_model.current_pipeline.save('pipeline_kick_name.json')
    print(f'R2 for validation sample is {round(r2, 3)}')

    return r2


if __name__ == '__main__':
    run_multi_modal_example(file_path='examples/data/jc_penney_products', is_visualise=True)
