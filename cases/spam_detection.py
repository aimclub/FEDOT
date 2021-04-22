import os
import tarfile

from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData, train_test_data_setup


def unpack_archived_data():
    archive_path = os.path.abspath(os.path.join('data', 'spamham.tar.gz'))
    if 'spamham' not in os.listdir(os.path.dirname(archive_path)):
        with tarfile.open(archive_path) as file:
            file.extractall(path=os.path.dirname(archive_path))
        print('Unpacking finished')


def execute_chain_for_text_problem(train_data, test_data):
    node_text_clean = PrimaryNode('text_clean')
    node_tfidf = SecondaryNode('tfidf', nodes_from=[node_text_clean])
    model_node = SecondaryNode('multinb', nodes_from=[node_tfidf])
    chain = Chain(model_node)
    chain.fit(train_data)

    predicted = chain.predict(test_data)

    roc_auc_metric = roc_auc(y_true=test_data.target, y_score=predicted.predict)

    return roc_auc_metric


def run_text_problem_from_meta_file():
    data_file_abspath = os.path.abspath(os.path.join('data', 'spam', 'spamham.csv'))

    data = InputData.from_text_meta_file(meta_file_path=data_file_abspath)

    train_data, test_data = train_test_data_setup(data, split_ratio=0.7)

    metric = execute_chain_for_text_problem(train_data, test_data)

    print(f'meta_file metric: {metric}')


def run_text_problem_from_files():
    unpack_archived_data()

    data_abspath = os.path.abspath(os.path.join('data', 'spamham'))

    train_path = os.path.join(data_abspath, 'train')
    test_path = os.path.join(data_abspath, 'test')

    train_data = InputData.from_text_files(files_path=train_path)
    test_data = InputData.from_text_files(files_path=test_path)

    metric = execute_chain_for_text_problem(train_data, test_data)

    print(f'origin files metric: {metric}')


def run_text_problem_from_saved_meta_file():
    data_file_abspath = os.path.abspath(os.path.join('data', 'spamham', 'meta_train.csv'))

    data = InputData.from_text_meta_file(meta_file_path=data_file_abspath)

    train_data, test_data = train_test_data_setup(data, split_ratio=0.7)

    metric = execute_chain_for_text_problem(train_data, test_data)

    print(f'meta_file metric: {metric}')


if __name__ == '__main__':
    run_text_problem_from_meta_file()
    run_text_problem_from_files()
    run_text_problem_from_saved_meta_file()
