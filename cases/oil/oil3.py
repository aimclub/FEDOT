from os.path import join as join
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from cases.oil.crm import CRMP as crm
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import fedot_project_root as project_root


def prepare_data(target_well=0, is_full=True, modify_target=True):
    filepath = join(project_root(), 'cases', 'data', 'oil')
    parse_date = True  # This dataset has dates instead of elapsed time. Hence convert to timedelta

    qi = pd.read_excel(join(filepath, 'injection.xlsx'), engine='openpyxl')
    qp = pd.read_excel(join(filepath, 'production.xlsx'), engine='openpyxl')
    percent_train = 0.7

    time_colname = 'Time [days]'
    if parse_date:
        qi[time_colname] = (qi.Date - qi.Date[0]) / pd.to_timedelta(1, unit='D')

    num_items = 40

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

    q_obs_train = q_obs[:n_train, :]
    q_obs_test = q_obs[n_train:, :]

    forecast_length = len(t_arr) - n_train

    ds_train = {}
    ds_test = {}

    task = Task(TaskTypesEnum.ts_forecasting,
                task_params=TsForecastingParams(forecast_length=forecast_length))

    idx_train = t_arr[:n_train]
    idx_test = t_arr[np.arange(n_train, n_train + forecast_length)]

    if is_full:
        for i in range(q_obs.shape[1]):
            ds_train[f'data_source_ts/prod_{i}'] = InputData(idx=idx_train,
                                                             features=q_obs_train[:, i],
                                                             target=q_obs_train[:, 0],
                                                             data_type=DataTypesEnum.ts,
                                                             task=task)

            ds_test[f'data_source_ts/prod_{i}'] = InputData(idx=idx_test,
                                                            features=q_obs_train[:, i],
                                                            target=q_obs_test[:forecast_length, 0],
                                                            data_type=DataTypesEnum.ts,
                                                            task=task)
    for i in range(qi_arr.shape[1]):
        ds_train[f'data_source_ts/inj_{i}'] = InputData(idx=idx_train,
                                                        features=qi_arr[:n_train, i],
                                                        target=q_obs_train,  # [:, 0],
                                                        data_type=DataTypesEnum.ts,
                                                        task=task)

        ds_test[f'data_source_ts/inj_{i}'] = InputData(idx=idx_test,
                                                       features=qi_arr[n_train:(n_train + forecast_length), i],
                                                       target=q_obs_test[:forecast_length, :],
                                                       data_type=DataTypesEnum.ts,
                                                       task=task)
    input_data_train = MultiModalData(ds_train)
    input_data_test = MultiModalData(ds_test)

    return input_data_train, input_data_test


def crm_fit(idx: np.array, features: np.array, target: np.array, params: dict):
    t_arr = idx
    input_series_train = [t_arr, features]

    q_obs_train = target

    n_inj = 5
    n_prd = 5

    tau = np.ones(n_prd)
    gain_mat = np.ones([n_inj, n_prd])
    gain_mat = gain_mat / (np.sum(gain_mat, 1).reshape([-1, 1]))
    qp0 = np.array([[0, 0, 0, 0, 0]])
    inputs_list = [tau, gain_mat, qp0]
    crm_model = crm(inputs_list, include_press=False)

    init_guess = inputs_list
    crm_model.fit_model(input_series=input_series_train,
                        q_obs=q_obs_train,
                        init_guess=init_guess)

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


def run_exp():
    for well_id in range(0, 5):
        train_data, test_data = prepare_data(well_id, is_full=True)
        train_data_full, test_data_full = prepare_data(well_id, is_full=True, modify_target=False)

        pipeline = get_simple_pipeline(train_data, well_id)

        pipeline.fit_from_scratch(train_data)
        pipeline.print_structure()

        predicted_test = pipeline.predict(test_data)

        train_data_full = train_data_full[f'data_source_ts/inj_0']
        test_data_full = test_data_full[f'data_source_ts/inj_0']

        plt.plot(np.append(train_data_full.idx, test_data_full.idx),
                 np.append(train_data_full.target[:, well_id],
                           test_data_full.target[:, well_id]),
                 linestyle='-', c='r', label='Actual')
        plt.plot(test_data_full.idx, np.ravel(predicted_test.predict),
                 linestyle='--', marker=None, c='b', label='Predicted')
        plt.title('Well ' + str(well_id), fontsize=15)
        plt.ylabel('Total Fluid (RB)', fontsize=13)
        plt.xlabel('Time (D)', fontsize=13)
        plt.legend(fontsize=11)
        plt.show()


if __name__ == '__main__':
    run_exp()
