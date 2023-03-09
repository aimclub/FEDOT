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
from fedot.core.utils import fedot_project_root


def get_pipeline():
    text_node = PrimaryNode('data_source_text/description')
    vect_node = SecondaryNode('tfidf', nodes_from=[text_node])
    model_node = SecondaryNode('logit', nodes_from=[vect_node])

    table_node = PrimaryNode('data_source_table')
    scaling_node = SecondaryNode('scaling', nodes_from=[table_node])
    model_node2 = SecondaryNode('logit', nodes_from=[scaling_node])

    final_node = SecondaryNode('logit', nodes_from=[model_node, model_node2])
    pipeline = Pipeline(final_node)
    return pipeline


def update_pipeline(pipeline: dict) -> Pipeline:
    final_pipeline = pipeline['data_source_table']
    pos_ens_node = len(final_pipeline.nodes)
    ensemble_node = SecondaryNode('logit')
    final_pipeline.add_node(ensemble_node)
    final_pipeline.connect_nodes(final_pipeline.nodes[0], ensemble_node)
    for node in pipeline['data_source_text/description'].nodes:
        final_pipeline.add_node(node)
    final_pipeline.connect_nodes(final_pipeline.nodes[pos_ens_node + 1], final_pipeline.nodes[pos_ens_node])
    return final_pipeline


def run_multi_modal_example(file_path: str, timeout=10, visualization=False) -> float:
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
    path = Path(fedot_project_root(), file_path)
    data = MultiModalData.from_csv(file_path=path, task=task, target_columns='variety', index_col=None)
    fit_data, predict_data = train_test_data_setup(data, shuffle_flag=True, split_ratio=0.7)
    pipelines_dict = {}
    timeout_source = int(timeout / (len(fit_data) + 1))

    for data_source in fit_data:
        fit_data_source = deepcopy(fit_data)
        predict_data_source = deepcopy(predict_data)
        sources_to_del = [el for el in list(fit_data.keys()) if el != data_source]
        for source in sources_to_del:
            del fit_data_source[source]
            del predict_data_source[source]

        automl_model = Fedot(problem=task, timeout=timeout_source, n_jobs=6, preset='best_quality')
        print(f'Fitting {data_source} for {timeout_source} minutes')
        automl_model.fit(features=fit_data_source,
                         target=fit_data_source.target)
        pipelines_dict[data_source] = automl_model.current_pipeline
        prediction = automl_model.predict(predict_data_source)
        metrics = automl_model.get_metrics()

    #if visualization:
        automl_model.current_pipeline.show(engine='pyvis')

        print(f'F1 for {data_source} is {round(metrics["f1"], 3)}')

    final_pipeline = update_pipeline(pipelines_dict)
    automl_model = Fedot(problem=task, timeout=timeout_source, n_jobs=6,
                         initial_assumption=final_pipeline, preset='best_quality', metric='f1')
    print(f'Fitting final model for {timeout_source} minutes')
    automl_model.fit(features=fit_data,
                     target=fit_data.target)
    prediction = automl_model.predict(predict_data)
    metrics = automl_model.get_metrics()
    automl_model.current_pipeline.show(engine='pyvis')
    automl_model.current_pipeline.save('pipeline_kick_name.json')
    print(f'F1 for final model is {round(metrics["f1"], 3)}')
    return metrics["f1"]


if __name__ == '__main__':
    # res = []
    # for i in range(5):
    metric = run_multi_modal_example(file_path='examples/data/multimodal_wine.csv', timeout=9, visualization=True)
    #     res.append(metric)
    # print(f'Average F1 is {round(sum(res) / len(res), 3)}')
