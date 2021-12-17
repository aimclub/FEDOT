from os.path import join as join
from typing import Any, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

from cases.oil.crm import CRMP
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import fedot_project_root as project_root


def prepare_data(target_well=0, modify_target=True,
                 parse_date=True, percent_train=0.7, num_items=40) -> Tuple[MultiModalData, MultiModalData]:
    """
    Prepares multi-modal dataset with exog. variables
    :param target_well: id of well for prediction
    :param modify_target: remove redundant columns from target
    :param parse_date: parse date from first column
    :param percent_train: ratio of train part
    :param num_items: size of dataset to use
    :return: train and test samples
    """
    filepath = join(project_root(), 'cases', 'data', 'oil')

    qi = pd.read_excel(join(filepath, 'injection.xlsx'), engine='openpyxl')
    qp = pd.read_excel(join(filepath, 'production.xlsx'), engine='openpyxl')

    time_colname = 'Time [days]'
    if parse_date:
        qi[time_colname] = (qi.Date - qi.Date[0]) / pd.to_timedelta(1, unit='D')

    inj_list = [x for x in qi.keys() if x.startswith('I')]
    prd_list = [x for x in qp.keys() if x.startswith('P')]
    t_arr = qi[time_colname].values[:num_items]

    qi_arr = qi[inj_list].values[:num_items, :]
    q_obs = qp[prd_list].values[:num_items, :]

    if modify_target:
        non_target_cols = list(range(q_obs.shape[1]))
        non_target_cols.remove(target_well)
        new_cols = [target_well] + non_target_cols
        q_obs = q_obs[:, new_cols]

    # Separation into training and test set
    n_train = round(percent_train * len(t_arr))

    forecast_length = len(t_arr) - n_train

    ds = {}
    task = Task(TaskTypesEnum.ts_forecasting,
                task_params=TsForecastingParams(forecast_length=forecast_length))

    idx = t_arr

    for i in range(q_obs.shape[1]):
        ds[f'data_source_ts/prod_{i}'] = InputData(idx=idx,
                                                   features=q_obs[:, i],
                                                   target=q_obs[:, 0],
                                                   data_type=DataTypesEnum.ts,
                                                   task=task)

    for i in range(qi_arr.shape[1]):
        ds[f'data_source_ts/inj_{i}'] = InputData(idx=idx,
                                                  features=qi_arr[:, i],
                                                  target=q_obs,
                                                  data_type=DataTypesEnum.ts,
                                                  task=task)

    input_data_train, input_data_test = train_test_data_setup(MultiModalData(ds), split_ratio=percent_train)
    return input_data_train, input_data_test


def crm_fit(idx: np.array, features: np.array, target: np.array, params: dict):
    t_arr = idx
    input_series_train = [t_arr, features]

    q_obs_train = target

    n_inj = features.shape[1]
    n_prd = q_obs_train.shape[1]

    tau = np.ones(n_prd)
    gain_mat = np.ones([n_inj, n_prd])
    gain_mat = gain_mat / (np.sum(gain_mat, 1).reshape([-1, 1]))
    qp0 = np.array([[0, 0, 0, 0, 0]])

    crm_model = CRMP(tau, gain_mat, qp0)
    crm_model.fit_model(input_series=input_series_train,
                        q_obs=q_obs_train,
                        tau_0=tau,
                        gain_mat_0=gain_mat,
                        q0_0=qp0)

    return crm_model


def crm_predict(fitted_model: Any, idx: np.array, features: np.array, params: dict):
    input_series_test = [idx, features]

    qp_pred_test = fitted_model.prod_pred(input_series=input_series_test)
    qp_pred_test = qp_pred_test[:, params['target_well_num']]
    return qp_pred_test, 'ts'


def get_simple_pipeline(multi_data, well_id):
    """
        Pipeline looking like this
        lagged -> custom -> ridge
    """

    inj_list = []
    prod_list = []

    for i, data_id in enumerate(multi_data.keys()):
        if 'inj_' in data_id:
            inj_list.append(PrimaryNode(data_id))
        if 'prod_' in data_id:
            lagged_node = SecondaryNode('lagged', nodes_from=[PrimaryNode(data_id)])
            lagged_node.custom_params = {'window_size': 12}

            prod_list.append(lagged_node)

    # For custom model params as initial approximation and model as function is necessary
    custom_node = SecondaryNode('custom/CRM', nodes_from=inj_list)
    custom_node.custom_params = {'model_predict': crm_predict,
                                 'model_fit': crm_fit,
                                 'target_well_num': well_id}

    exog_pred_node = SecondaryNode('exog_ts', nodes_from=[custom_node])

    final_ens = [exog_pred_node] + prod_list
    node_final = SecondaryNode('ridge', nodes_from=final_ens)
    pipeline = Pipeline(node_final)
    pipeline.show()

    return pipeline


def plot_lines(well_id, train_data_full, test_data_full, predicted_test):
    rmse = mean_squared_error(test_data_full.target[:, well_id],
                              np.ravel(predicted_test.predict), squared=False)

    plt.plot(np.append(train_data_full.idx, test_data_full.idx),
             np.append(train_data_full.target[:, well_id],
                       test_data_full.target[:, well_id]),
             linestyle='-', c='r', label='Actual')
    plt.plot(test_data_full.idx, np.ravel(predicted_test.predict),
             linestyle='--', marker=None, c='b', label='Predicted')
    plt.title(f'Well {str(well_id)}, MSE: {round(rmse, 2)}', fontsize=15)
    plt.ylabel('Total Fluid (RB)', fontsize=13)
    plt.xlabel('Time (D)', fontsize=13)
    plt.legend(fontsize=11)
    plt.show()

    print(rmse)


def run_oil_pipeline():
    for well_id in range(0, 5):
        train_data, test_data = prepare_data(well_id)
        train_data_full, test_data_full = prepare_data(well_id, modify_target=False)

        pipeline = get_simple_pipeline(train_data, well_id)

        pipeline.fit_from_scratch(train_data)
        pipeline.print_structure()

        predicted_test = pipeline.predict(test_data)

        train_data_full = train_data_full[f'data_source_ts/inj_0']
        test_data_full = test_data_full[f'data_source_ts/inj_0']

        plot_lines(well_id, train_data_full, test_data_full, predicted_test)


if __name__ == '__main__':
    run_oil_pipeline()
