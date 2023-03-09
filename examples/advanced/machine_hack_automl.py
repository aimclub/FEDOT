from sklearnex import patch_sklearn
patch_sklearn()
from pathlib import Path

from fedot.api.main import Fedot

from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.utils import fedot_project_root

from sklearn.metrics import r2_score, roc_auc_score


def word2vec_pipeline():
    text_node1 = PrimaryNode('data_source_text/question_title')
    word2vec_node1 = SecondaryNode('word2vec_pretrained', nodes_from=[text_node1])
    text_scaling_node1 = SecondaryNode('scaling', nodes_from=[word2vec_node1])
    text_rf1 = SecondaryNode('rfr', nodes_from=[text_scaling_node1])

    text_node2 = PrimaryNode('data_source_text/question_body')
    word2vec_node2 = SecondaryNode('word2vec_pretrained', nodes_from=[text_node2])
    text_scaling_node2 = SecondaryNode('scaling', nodes_from=[word2vec_node2])
    text_rf2 = SecondaryNode('rfr', nodes_from=[text_scaling_node2])

    text_node3 = PrimaryNode('data_source_text/answer')
    word2vec_node3 = SecondaryNode('word2vec_pretrained', nodes_from=[text_node3])
    text_scaling_node3 = SecondaryNode('scaling', nodes_from=[word2vec_node3])
    text_rf3 = SecondaryNode('rfr', nodes_from=[text_scaling_node3])

    table_node = PrimaryNode('data_source_table')
    table_scaling_node = SecondaryNode('scaling', nodes_from=[table_node])
    table_rf = SecondaryNode('rfr', nodes_from=[table_scaling_node])
    final_node = SecondaryNode('rfr', nodes_from=[text_rf1, text_rf2, text_rf3, table_rf])
    return Pipeline(final_node)


def run_machine_hack_sentiment_analysis(file_path: str, is_visualise=True,
                                        timeout: int = 10) -> float:
    """
    This is an example of FEDOT use on multimodal data.
    The data is taken from Jigsaw 2019 Kaggle competition:
    https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification
    and contains information about the post (e.g. likes, rating, date created, etc.).
    Column that contains text features is 'created_date'.
    Other columns contain numerical and categorical features.
    The task is to predict whether online social media comments are toxic
    based on their text and additional tabular features.

    :param file_path: path to the file with multimodal data
    :param is_visualise: if True, then final pipeline will be visualised
    :param timeout: time limit in seconds for FEDOT training

    :return: F1 metrics of the model
    """
    task = 'classification'
    text_columns = ['question_title', 'question_body', 'answer']
    columns_to_drop = ['created_date']
    train_path = Path(fedot_project_root(), file_path, 'jigsaw_unintended_bias100k_train.csv')
    test_path = Path(fedot_project_root(), file_path, 'jigsaw_unintended_bias100k_test.csv')
    fit_data = MultiModalData.from_csv(file_path=train_path, task=task,
                                       columns_to_drop=columns_to_drop,
                                       # text_columns=text_columns,
                                       target_columns='target')
    predict_data = MultiModalData.from_csv(file_path=test_path, task=task,
                                           columns_to_drop=columns_to_drop,
                                           # text_columns=text_columns,
                                           target_columns='target')

    # w2v_pipeline = word2vec_pipeline()
    automl_model = Fedot(problem=task, timeout=timeout, n_jobs=6, with_tuning=False, safe_mode=False,
                         preset='best_quality')
    automl_model.fit(features=fit_data,
                     target=fit_data.target,
                     predefined_model='auto')

    prediction = automl_model.predict(predict_data)
    metrics = automl_model.get_metrics()

    if is_visualise:
        automl_model.current_pipeline.show()
    # automl_model.history.show()
    roc_auc = metrics["roc_auc"]
    print(f'ROC-AUC for validation sample is {round(roc_auc, 3)}')
    return roc_auc


if __name__ == '__main__':
    run_machine_hack_sentiment_analysis(file_path='examples/data/jigsaw_unintended_bias100k/', is_visualise=True, timeout=10)
