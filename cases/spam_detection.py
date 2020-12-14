import os

import nltk
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.composer.chain import Chain
from fedot.core.composer.node import PrimaryNode
from fedot.core.models.data import InputData, train_test_data_setup
from fedot.core.models.preprocessing import EmptyStrategy


def run_text_problem(data_path):
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    data = InputData.from_text(meta_file=data_path, lang='english')

    train_data, test_data = train_test_data_setup(data, split_ratio=0.7)

    node = PrimaryNode('multinb')
    node.manual_preprocessing_func = EmptyStrategy
    chain = Chain()
    chain.add_node(node)
    chain.fit(train_data)

    predicted = chain.predict(test_data)

    roc_auc_metric = roc_auc(y_true=test_data.target, y_score=predicted.predict)

    print(roc_auc_metric)


if __name__ == '__main__':
    data_file = os.path.join('spam', 'spamham.csv')
    data_file_abspath = os.path.abspath(data_file)
    run_text_problem(data_path=data_file_abspath)
