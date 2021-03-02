import datetime
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.operations.tuning.hyperopt_tune.entire_tuning import ChainTuner
from fedot.core.operations.tuning.hyperopt_tune.node_tuning import NodesTuner

np.random.seed(10)


def run_experiment(file_path, chain):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
        np.array(df[['level_station_1', 'month', 'mean_temp', 'precip']]),
        np.array(df['level_station_2']),
        test_size=0.2,
        shuffle=True,
        random_state=10)

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
    first_mae = mean_absolute_error(y_data_test, preds)
    print(f'MAE before tuning - {first_mae:.2f}')

    chain_tuner = NodesTuner(chain=chain, task=task,
                             max_lead_time=datetime.timedelta(minutes=2),
                             iterations=8)
    chain = chain_tuner.tune(input_data=train_input)

    # Fit it
    chain.fit_from_scratch(train_input)

    # Predict
    predicted_values = chain.predict(predict_input)
    preds = predicted_values.predict

    y_data_test = np.ravel(y_data_test)
    second_mae = mean_absolute_error(y_data_test, preds)
    print(f'MAE after tuning - {second_mae:.2f}\n')


if __name__ == '__main__':
    node_encoder = PrimaryNode('one_hot_encoding')

    # Parameters
    node_rans = SecondaryNode('ransac_lin_reg', nodes_from=[node_encoder])
    node_scaling = SecondaryNode('scaling', nodes_from=[node_rans])
    node_final = SecondaryNode('rfr', nodes_from=[node_scaling])
    chain = Chain(node_final)

    run_experiment('../cases/data/river_levels/station_levels.csv', chain)









