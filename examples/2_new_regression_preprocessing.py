import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from fedot.core.data.preprocessing import preprocessing_func_for_data, PreprocessingStrategy, \
    Scaling, Normalization, ImputationStrategy, EmptyStrategy
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.utilities.synthetic.data import regression_dataset


np.random.seed(2020)

################################################################################
#                                  ОПИСАНИЕ                                    #
################################################################################
"""
Данный скрипт запускает простой пример с генерацией датасета для задачи 
регрессии. В качестве построенной модели используется простая цепочка из одной
модели. Операции препрцессинга - новые, введен новый класс с потомками DataOperation.

При запуске файла 1_simple_regression_preprocessing.py (расположен в той же папке
проекта) ответ должен получиться идентичным для той же стратегии препроцессинга.
 - В таком случае логика исходных процедур не нарушена.
"""
################################################################################
#                                  ОПИСАНИЕ                                    #
################################################################################

def prepare_regression_dataset(samples_amount=250, features_amount=5,
                               features_options={'informative': 2,'bias': 1.0}):
    """
    Prepares four numpy arrays with different scale features and target
    :param samples_amount: Total amount of samples in the resulted dataset.
    :param features_amount: Total amount of features per sample.
    :param features_options: The dictionary containing features options in key-value
    format:
        - informative: the amount of informative features;
        - bias: bias term in the underlying linear model;
    :return x_data_train: features to train
    :return y_data_train: target to train
    :return x_data_test: features to test
    :return y_data_test: target to test
    """

    x_data, y_data = regression_dataset(samples_amount=samples_amount,
                                        features_amount=features_amount,
                                        features_options=features_options,
                                        n_targets=1,
                                        noise=0.0, shuffle=True)

    # Changing the scale of the data
    for i, coeff in zip(range(0, features_amount),
                        np.random.randint(1,100,features_amount)):
        # Get column
        feature = np.array(x_data[:, i])

        # Change scale for this feature
        rescaled = feature * coeff
        x_data[:, i] = rescaled

    # Train and test split
    x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(x_data, y_data,
                                                                            test_size = 0.3)

    return x_data_train, y_data_train, x_data_test, y_data_test

if __name__ == '__main__':
    x_data_train, y_data_train, \
    x_data_test, y_data_test = prepare_regression_dataset(150,3,{'informative': 2,'bias': 1.0})

    # Interface for such data operation for now look like this
    node_scaling = PrimaryNode('scaling', manual_preprocessing_func=EmptyStrategy)
    node_final = SecondaryNode('ridge', manual_preprocessing_func=EmptyStrategy,
                               nodes_from=[node_scaling])
    chain = Chain(node_final)

    # Define regression task
    task = Task(TaskTypesEnum.regression)

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(x_data_train)),
                            features=x_data_train,
                            target=y_data_train,
                            task=task,
                            data_type=DataTypesEnum.table)

    predict_input = InputData(idx=np.arange(0, len(x_data_test)),
                              features=x_data_test,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.table)
    # Fit it
    chain.fit(train_input, verbose=True)
    # Predict
    predicted_values = chain.predict(predict_input)
    preds = predicted_values.predict

    y_data_test = np.ravel(y_data_test)
    print(f'Предсказанные значения: {preds[:5]}')
    print(f'Действительные значения: {y_data_test[:5]}')
    print(f'RMSE - {mean_squared_error(y_data_test, preds, squared=False):.2f}\n')