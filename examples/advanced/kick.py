import os
import sys
from copy import deepcopy
from pathlib import Path
from sklearnex import patch_sklearn

from fedot.core.repository.dataset_types import DataTypesEnum

patch_sklearn()
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


AV_OPS_TEXT = ['resample', 'pca', 'lgbm', 'knn', 'bernb', 'qda', 'scaling',
               'mlp', 'poly_features', 'logit', 'normalization', 'fast_ica', 'tfidf', 'cntvect']

AV_OPS_TABLE = ['resample', 'pca', 'lgbm', 'isolation_forest_class', 'knn', 'bernb', 'qda', 'scaling',
               'mlp', 'poly_features', 'dt', 'logit', 'normalization', 'fast_ica', 'tfidf', 'cntvect', 'rf']


def get_best_pipeline():
    # Acc = 0.
    tbl_node = PrimaryNode('data_source_table')
    ridge_node_tbl = SecondaryNode('lgbm', nodes_from=[tbl_node])  # 0.618 lgbm
    ridge_node_tbl.custom_params = {
        "num_leaves": 10,
        "colsample_bytree": 0.6113129589880463,
        "subsample": 0.6893882145276637,
        "subsample_freq": 10,
        "learning_rate": 0.1990749007811331,
        "n_estimators": 100,
        "class_weight": "balanced",
        "reg_alpha": 0.11368459260078029,
        "reg_lambda": 4.248840300141813e-08
    }

    text_node0 = PrimaryNode('data_source_text/name')
    vect_node0 = SecondaryNode('tfidf', nodes_from=[text_node0])
    vect_node0.custom_params = {
        "min_df": 0.000587124514537694,
        "max_df": 0.9518506021961133,
        "max_features": 100000,
        "ngram_range": [1, 4]
    }
    ridge_node0 = SecondaryNode('rf', nodes_from=[vect_node0])  # 0.522 / 0.559 rf

    text_node1 = PrimaryNode('data_source_text/desc')
    vect_node1 = SecondaryNode('cntvect', nodes_from=[text_node1])
    vect_node1.custom_params = {'ngram_range': (1, 2)}
    ridge_node1 = SecondaryNode('ridge', nodes_from=[vect_node1])  # 0.

    text_node2 = PrimaryNode('data_source_text/keywords')
    vect_node2 = SecondaryNode('cntvect', nodes_from=[text_node2])
    vect_node2.custom_params = {'ngram_range': (1, 3)}
    ridge_node2 = SecondaryNode('ridge', nodes_from=[vect_node2])  # 0.

    final_node = SecondaryNode('ridge', nodes_from=[ridge_node_tbl, ridge_node0])
    pipeline = Pipeline(ridge_node0)
    return pipeline


def get_pipeline(fit_data: MultiModalData):
    if fit_data.data_type[0] is DataTypesEnum.text:
        primary_node = PrimaryNode(list(fit_data.keys())[0])
        vect_node = SecondaryNode('tfidf', nodes_from=[primary_node])
        model_node = SecondaryNode('logit', nodes_from=[vect_node])
    else:
        primary_node = PrimaryNode('data_source_table')
        scaling_node = SecondaryNode('scaling', nodes_from=[primary_node])
        model_node = SecondaryNode('logit', nodes_from=[scaling_node])
    return Pipeline(model_node)


def update_pipeline(pipeline: dict) -> Pipeline:
    final_pipeline = pipeline['data_source_table']
    pos_ens_node = len(final_pipeline.nodes)
    ensemble_node = SecondaryNode('logit')
    final_pipeline.add_node(ensemble_node)
    final_pipeline.connect_nodes(final_pipeline.nodes[0], ensemble_node)
    for node in pipeline['data_source_text/name'].nodes:
        final_pipeline.add_node(node)
    final_pipeline.connect_nodes(final_pipeline.nodes[pos_ens_node + 1], final_pipeline.nodes[pos_ens_node])
    for node in pipeline['data_source_text/desc'].nodes:
        final_pipeline.add_node(node)
    final_pipeline.connect_nodes(final_pipeline.nodes[pos_ens_node
                                                      + len(pipeline['data_source_text/name'].nodes) + 1],
                                 final_pipeline.nodes[pos_ens_node])
    for node in pipeline['data_source_text/keywords'].nodes:
        final_pipeline.add_node(node)
    final_pipeline.connect_nodes(final_pipeline.nodes[pos_ens_node
                                                      + len(pipeline['data_source_text/name'].nodes)
                                                      + len(pipeline['data_source_text/desc'].nodes)+ 1],
                                 final_pipeline.nodes[pos_ens_node])
    # final_pipeline.save(f'pipeline_fake', is_datetime_in_path=True)
    return final_pipeline


def load_pipeline(path: str):
    # 0.
    # path = create_correct_path(path)
    pipeline = Pipeline().load(path)
    return pipeline


def run_multi_modal_example(file_path: str, is_visualise=True, timeout=10) -> float:
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
    task = 'classification'
    train_path = Path(fedot_project_root(), file_path, 'kick_starter_funding_train.csv')
    test_path = Path(fedot_project_root(), file_path, 'kick_starter_funding_test.csv')
    fit_data = MultiModalData.from_csv(file_path=train_path, task=task, index_col=None,
                                       text_columns=['name', 'desc', 'keywords'],
                                       target_columns='final_status')
    predict_data = MultiModalData.from_csv(file_path=test_path, task=task, index_col=None,
                                           text_columns=['name', 'desc', 'keywords'],
                                           target_columns='final_status')

    pipelines_dict = {}
    timeout_source = int(timeout / len(fit_data))

    for data_source in fit_data:
        fit_data_source = deepcopy(fit_data)
        predict_data_source = deepcopy(predict_data)
        sources_to_del = [el for el in list(fit_data.keys()) if el != data_source]
        for source in sources_to_del:
            del fit_data_source[source]
            del predict_data_source[source]


        automl_model = Fedot(problem=task, timeout=timeout_source, n_jobs=6,
                             safe_mode=False, available_operations=AV_OPS_TEXT, metric='roc_auc',
                             initial_assumption=get_pipeline(fit_data_source))
        print(f'Fitting {data_source} for {timeout_source} minutes')
        automl_model.fit(features=fit_data_source,
                         target=fit_data_source.target)
        pipelines_dict[data_source] = automl_model.current_pipeline
        prediction = automl_model.predict(predict_data_source)
        metrics = automl_model.get_metrics(metric_names=['acc', 'f1', 'roc_auc'])

    #if visualization:
        automl_model.current_pipeline.show(engine='pyvis')
        automl_model.current_pipeline.save(f'pipeline_kick_{data_source}', is_datetime_in_path=True)
        history_name = data_source.replace('/', '_')
        automl_model.history.save(f'pipeline_kick/pipeline_kick_{history_name}.json')
        print(f'ROC-AUC for {data_source} is {round(metrics["roc_auc"], 3)}')

    # path = 'examples/advanced/December-15-2022,18-22-57,PM pipeline_kick_name'
    # path = 'examples/advanced/December-15-2022,18-18-12,PM pipeline_kick_name'
    final_pipeline = update_pipeline(pipelines_dict)
    automl_model = Fedot(problem=task, timeout=timeout, n_jobs=6, safe_mode=False,
                         initial_assumption=final_pipeline, available_operations=AV_OPS_TEXT, metric='roc_auc')
    print(f'Fitting final model for {timeout} minutes')
    automl_model.fit(features=fit_data,
                     target=fit_data.target)
    prediction = automl_model.predict(predict_data)
    metrics = automl_model.get_metrics(metric_names=['acc', 'f1', 'roc_auc'])
    automl_model.current_pipeline.show(engine='pyvis')
    roc_auc = round(metrics["roc_auc"], 3)
    automl_model.current_pipeline.save(f'pipeline_kick', is_datetime_in_path=True)
    automl_model.history.save(f'pipeline_kick/pipeline_kick.json')
    print(f'ROC-AUC for final model is {roc_auc}')
    return metrics["roc_auc"]


if __name__ == '__main__':
    #TODO: try trees
    metric = run_multi_modal_example(file_path='examples/data/kick_starter_funding', is_visualise=True, timeout=30)
