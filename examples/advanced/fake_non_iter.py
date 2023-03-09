from copy import deepcopy
from pathlib import Path

from fedot.api.api_utils.assumptions.assumptions_builder import UniModalAssumptionsBuilder
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.utils import fedot_project_root

AV_OPS_TEXT = ['resample', 'pca', 'lgbm', 'knn', 'bernb', 'qda', 'scaling',
               'mlp', 'poly_features', 'logit', 'normalization', 'fast_ica', 'tfidf', 'cntvect']

AV_OPS_TABLE = ['resample', 'pca', 'lgbm', 'isolation_forest_class', 'knn', 'bernb', 'qda', 'scaling',
               'mlp', 'poly_features', 'dt', 'logit', 'normalization', 'fast_ica', 'tfidf', 'cntvect', 'rf']

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


def get_final_pipeline():
    table_node = PrimaryNode('data_source_table')
    scaling_node = SecondaryNode('scaling', nodes_from=[table_node])
    model_node = SecondaryNode('logit', nodes_from=[scaling_node])

    text_node_title = PrimaryNode('data_source_text/title')
    vect_node_title = SecondaryNode('tfidf', nodes_from=[text_node_title])
    model_node_title = SecondaryNode('logit', nodes_from=[vect_node_title])

    text_node_description = PrimaryNode('data_source_text/description')
    vect_node_description = SecondaryNode('tfidf', nodes_from=[text_node_description])
    model_node_description = SecondaryNode('logit', nodes_from=[vect_node_description])

    ensemble_node = SecondaryNode('logit', nodes_from=[model_node, model_node_title, model_node_description])
    return Pipeline(ensemble_node)


def update_pipeline(pipeline: dict) -> Pipeline:
    final_pipeline = pipeline['data_source_table']
    pos_ens_node = len(final_pipeline.nodes)
    ensemble_node = SecondaryNode('logit')
    final_pipeline.add_node(ensemble_node)
    final_pipeline.connect_nodes(final_pipeline.nodes[0], ensemble_node)
    for node in pipeline['data_source_text/title'].nodes:
        final_pipeline.add_node(node)
    final_pipeline.connect_nodes(final_pipeline.nodes[pos_ens_node + 1], final_pipeline.nodes[pos_ens_node])
    for node in pipeline['data_source_text/description'].nodes:
        final_pipeline.add_node(node)
    final_pipeline.connect_nodes(final_pipeline.nodes[pos_ens_node + len(pipeline['data_source_text/title'].nodes) + 1],
                                 final_pipeline.nodes[pos_ens_node])
    final_pipeline.save(f'pipeline_fake', is_datetime_in_path=True)
    return final_pipeline


def run_multi_modal_example(file_path: str, timeout=10, visualization=False, i=0) -> float:
    """
    This is an example of FEDOT use on multimodal data.
    The data is taken and adapted from Wine Reviews dataset (winemag-data_first150k):
    https://www.kaggle.com/datasets/zynicide/wine-reviews
    and contains information about wine country, region, price, etc.
    Column that contains text features is 'description'.
    Other columns contain numerical and categorical features.
    The aim is to predict wine variety, so it's a classification task.

    :param file_path: path to the file with multimodal data
    :param timeout: time limit for pipeline search
    :param visualization: if True, then final pipeline will be visualised

    :return: F1 metrics of the model
    """
    task = 'classification'
    train_path = Path(fedot_project_root(), file_path, 'fake_job_postings2_train.csv')
    test_path = Path(fedot_project_root(), file_path, 'fake_job_postings2_test.csv')
    fit_data = MultiModalData.from_csv(file_path=train_path, task=task, index_col=None,
                                       text_columns=['title', 'description'],
                                       columns_to_drop=['salary_range'],
                                       target_columns='fraudulent')
    predict_data = MultiModalData.from_csv(file_path=test_path, task=task, index_col=None,
                                           text_columns=['title', 'description'],
                                           columns_to_drop=['salary_range'],
                                           target_columns='fraudulent')
    pipelines_dict = {}
    timeout_source = int(timeout / (len(fit_data) + 1))

    # for data_source in fit_data:
    #     fit_data_source = deepcopy(fit_data)
    #     predict_data_source = deepcopy(predict_data)
    #     sources_to_del = [el for el in list(fit_data.keys()) if el != data_source]
    #     for source in sources_to_del:
    #         del fit_data_source[source]
    #         del predict_data_source[source]
    #
    #
    #     automl_model = Fedot(problem=task, timeout=timeout_source, n_jobs=6,
    #                          safe_mode=False, available_operations=AV_OPS_TEXT, metric='roc_auc',
    #                          initial_assumption=get_pipeline(fit_data_source))
    #     print(f'Fitting {data_source} for {timeout_source} minutes')
    #     automl_model.fit(features=fit_data_source,
    #                      target=fit_data_source.target)
    #     pipelines_dict[data_source] = automl_model.current_pipeline
    #     prediction = automl_model.predict(predict_data_source)
    #     metrics = automl_model.get_metrics(metric_names=['acc', 'f1', 'roc_auc'])
    #
    # #if visualization:
    #     # automl_model.current_pipeline.show(engine='pyvis')
    #
    #     print(f'ROC-AUC for {data_source} is {round(metrics["roc_auc"], 3)}')

    # final_pipeline = update_pipeline(pipelines_dict)
    automl_model = Fedot(problem=task, timeout=timeout, n_jobs=2, safe_mode=False,
                         # initial_assumption=final_pipeline,
                         initial_assumption=get_final_pipeline(),
                         available_operations=AV_OPS_TEXT, metric='roc_auc')
    print(f'Fitting final model for {timeout} minutes')
    automl_model.fit(features=fit_data,
                     target=fit_data.target)
    prediction = automl_model.predict(predict_data)
    metrics = automl_model.get_metrics(metric_names=['acc', 'f1', 'roc_auc'])
    automl_model.current_pipeline.show(engine='pyvis')
    roc_auc = round(metrics["roc_auc"], 3)
    automl_model.current_pipeline.save(f'pipeline_fake_non_iter', is_datetime_in_path=True)
    automl_model.history.save(f'pipeline_fake_non_iter/pipeline_fake_non_iter_{i}.json')
    print(f'ROC-AUC for final model is {roc_auc}')
    return metrics["roc_auc"]


if __name__ == '__main__':
    res = []
    for i in range(5):
        print(f'Iteration {i + 1}')
        metric = run_multi_modal_example(file_path='examples/data/fake_job_postings2', timeout=30, visualization=True, i=i)
        res.append(metric)
    print(f'Average F1 is {round(sum(res) / len(res), 3)}')
