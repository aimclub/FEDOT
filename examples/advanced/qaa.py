import os
import sys
from copy import deepcopy
from pathlib import Path
from sklearnex import patch_sklearn

from fedot.core.repository.dataset_types import DataTypesEnum

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


AV_OPS_TEXT = ['adareg', 'gbr', 'knnreg', 'lasso', 'lgbmreg', 'linear', 'ridge',
               'sgdr', 'svr', 'scaling', 'normalization', 'pca', 'fast_ica', 'poly_features',
               'ransac_lin_reg', 'ransac_non_lin_reg', 'tfidf', 'cntvect']

AV_OPS_TABLE = ['adareg', 'dtreg', 'gbr', 'knnreg', 'lasso', 'lgbmreg', 'linear', 'rfr', 'ridge',
               'sgdr', 'svr', 'treg', 'scaling', 'normalization', 'pca', 'fast_ica', 'poly_features',
               'ransac_lin_reg', 'ransac_non_lin_reg', 'isolation_forest_reg', 'tfidf', 'cntvect']

columns_to_drop = ['qa_id', 'question_user_name', 'question_user_page',
                   'answer_user_name', 'answer_user_page', 'url',
                   'host', 'question_asker_intent_understanding',
                   'question_body_critical', 'question_conversational',
                   'question_expect_short_answer', 'question_fact_seeking',
                   'question_has_commonly_accepted_answer', 'question_interestingness_others',
                   'question_interestingness_self', 'question_multi_intent',
                   'question_not_really_a_question', 'question_opinion_seeking',
                   'question_type_choice', 'question_type_compare',
                   'question_type_consequence', 'question_type_definition',
                   'question_type_entity', 'question_type_instructions',
                   'question_type_procedure', 'question_type_reason_explanation',
                   'question_type_spelling', 'question_well_written',
                   'answer_helpful', 'answer_level_of_information',
                   'answer_plausible', 'answer_relevance', 'answer_satisfaction',
                   'answer_type_instructions', 'answer_type_procedure',
                   'answer_well_written']


def get_pipeline(fit_data: MultiModalData):
    if fit_data.data_type[0] is DataTypesEnum.text:
        node_name = list(fit_data.keys())[0]
        primary_node = PrimaryNode(node_name)
        vect_node = SecondaryNode('tfidf', nodes_from=[primary_node])
        model_node = SecondaryNode('ridge', nodes_from=[vect_node])
    else:
        primary_node = PrimaryNode('data_source_table')
        scaling_node = SecondaryNode('scaling', nodes_from=[primary_node])
        model_node = SecondaryNode('ridge', nodes_from=[scaling_node])
    return Pipeline(model_node)


def update_pipeline(pipeline: dict) -> Pipeline:
    final_pipeline = pipeline['data_source_table']
    pos_ens_node = len(final_pipeline.nodes)
    ensemble_node = SecondaryNode('ridge')
    final_pipeline.add_node(ensemble_node)
    final_pipeline.connect_nodes(final_pipeline.nodes[0], ensemble_node)
    for node in pipeline['data_source_text/question_title'].nodes:
        final_pipeline.add_node(node)
    final_pipeline.connect_nodes(final_pipeline.nodes[pos_ens_node + 1], final_pipeline.nodes[pos_ens_node])
    for node in pipeline['data_source_text/question_body'].nodes:
        final_pipeline.add_node(node)
    final_pipeline.connect_nodes(final_pipeline.nodes[pos_ens_node
                                                      + len(pipeline['data_source_text/question_title'].nodes) + 1],
                                 final_pipeline.nodes[pos_ens_node])
    for node in pipeline['data_source_text/answer'].nodes:
        final_pipeline.add_node(node)
    final_pipeline.connect_nodes(final_pipeline.nodes[pos_ens_node
                                                      + len(pipeline['data_source_text/question_title'].nodes)
                                                      + len(pipeline['data_source_text/question_body'].nodes) + 1],
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
    task = 'regression'
    train_path = Path(fedot_project_root(), file_path, 'google_qa_answer_type_reason_explanation_train.csv')
    test_path = Path(fedot_project_root(), file_path, 'google_qa_answer_type_reason_explanation_test.csv')
    fit_data = MultiModalData.from_csv(file_path=train_path, task=task, index_col=None,
                                       columns_to_drop=columns_to_drop,
                                       text_columns=['question_title', 'question_body', 'answer'],
                                       target_columns='answer_type_reason_explanation')
    predict_data = MultiModalData.from_csv(file_path=test_path, task=task, index_col=None,
                                           columns_to_drop=columns_to_drop,
                                           text_columns=['question_title', 'question_body', 'answer'],
                                           target_columns='answer_type_reason_explanation')

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
                             safe_mode=False,
                             available_operations=AV_OPS_TEXT,
                             metric='r2',
                             initial_assumption=get_pipeline(fit_data_source))
        print(f'Fitting {data_source} for {timeout_source} minutes')
        automl_model.fit(features=fit_data_source,
                         target=fit_data_source.target)
        pipelines_dict[data_source] = automl_model.current_pipeline
        prediction = automl_model.predict(predict_data_source)
        metrics = automl_model.get_metrics(metric_names=['r2'])

    #if visualization:
        automl_model.current_pipeline.show(engine='pyvis')
        automl_model.current_pipeline.save(f'pipeline_qaa/pipeline_qaa_{data_source}', is_datetime_in_path=True)
        history_name = data_source.replace('/', '_')
        automl_model.history.save(f'pipeline_qaa/pipeline_qaa_{history_name}.json')
        print(f'R2 for {data_source} is {round(metrics["r2"], 3)}')
        print(f'R2 true for {data_source} is {round(r2_score(predict_data_source.target, prediction), 3)}')


    final_pipeline = update_pipeline(pipelines_dict)
    automl_model = Fedot(problem=task, timeout=timeout, n_jobs=6, safe_mode=False,
                         initial_assumption=final_pipeline,
                         available_operations=AV_OPS_TEXT,
                         metric='r2')
    print(f'Fitting final model for {timeout} minutes')
    automl_model.fit(features=fit_data,
                     target=fit_data.target)
    prediction = automl_model.predict(predict_data)
    metrics = automl_model.get_metrics(metric_names=['r2'])
    automl_model.current_pipeline.show(engine='pyvis')
    r2 = round(metrics["r2"], 3)
    automl_model.current_pipeline.save(f'pipeline_qaa', is_datetime_in_path=True)
    automl_model.history.save(f'pipeline_qaa/pipeline_qaa.json')
    print(f'R2 for final model is {r2}')
    print(f'R2 true for final model is {round(r2_score(predict_data.target, prediction), 3)}')
    return metrics["r2"]


if __name__ == '__main__':
    metric = run_multi_modal_example(file_path='examples/data/google_qa_answer_type_reason_explanation',
                                     is_visualise=True, timeout=60)
