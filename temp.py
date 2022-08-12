import os

import numpy as np

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


if __name__ == '__main__':
    train_data_path = 'C:/Users/andre/Documents/GitHub/Huawei_AutoML/data/classification/train_sylvine_fold0.npy'
    train_labels_path = 'C:/Users/andre/Documents/GitHub/Huawei_AutoML/data/classification/trainy_sylvine_fold0.npy'
    test_data_path = 'C:/Users/andre/Documents/GitHub/Huawei_AutoML/data/classification/test_sylvine_fold0.npy'
    test_labels_path = 'C:/Users/andre/Documents/GitHub/Huawei_AutoML/data/classification/testy_sylvine_fold0.npy'

    auto_model = Fedot(problem='classification')

    train = np.load(train_data_path, allow_pickle=True)
    train_labels = np.load(train_labels_path, allow_pickle=True)
    test = np.load(test_data_path, allow_pickle=True)
    test_labels = np.load(test_labels_path, allow_pickle=True)

    pipeline = PipelineBuilder().add_sequence('poly_features', 'rf').to_pipeline()
    auto_model.fit(features=train, target=train_labels, predefined_model=pipeline)
    prediction = auto_model.predict(features=test)

    print(auto_model.get_metrics(test_labels))
    auto_model.plot_prediction()




