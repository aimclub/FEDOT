import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from fedot.core.data.preprocessing import preprocessing_func_for_data, PreprocessingStrategy, \
    Scaling, Normalization, ImputationStrategy, EmptyStrategy
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.utilities.synthetic.data import classification_dataset

np.random.seed(2020)


def prepare_classification_dataset(samples_amount=250, features_amount=5,
                                   classes_amount=2,
                                   features_options={'informative': 2,
                                                     'redundant': 2,
                                                     'repeated': 1,
                                                     'clusters_per_class': 1}):
    """
    Prepares four numpy arrays with different scale features and target
    :param samples_amount: Total amount of samples in the resulted dataset.
    :param features_amount: Total amount of features per sample.
    :param classes_amount: The amount of classes in the dataset.
    :param features_options: The dictionary containing features options in key-value
    format:
        - informative: the amount of informative features;
        - redundant: the amount of redundant features;
        - repeated: the amount of features that repeat the informative features;
        - clusters_per_class: the amount of clusters for each class;
    :return x_data_train: features to train
    :return y_data_train: target to train
    :return x_data_test: features to test
    :return y_data_test: target to test
    """

    x_data, y_data = classification_dataset(samples_amount=samples_amount,
                                            features_amount=features_amount,
                                            classes_amount=classes_amount,
                                            features_options=features_options)

    # Changing the scale of the data
    for i, coeff in zip(range(0, features_amount),
                        np.random.randint(1,100,features_amount)):
        # Get column
        feature = np.array(x_data[:, i])

        # Change scale for this feature
        rescaled = feature * coeff
        x_data[:, i] = rescaled

    # Train and test split
    x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(x_data, y_data,
                                                                            test_size=0.3)

    return x_data_train, y_data_train, x_data_test, y_data_test


# Do we need to process data with single-model chain
single = False
model = 'logit'
if __name__ == '__main__':
    samples = [50, 550, 150]
    features = [1, 5, 10]
    classes = [2, 2, 2]
    options = [{'informative': 1,'redundant': 0,'repeated': 0,'clusters_per_class': 1},
               {'informative': 2,'redundant': 1,'repeated': 1,'clusters_per_class': 1},
               {'informative': 3,'redundant': 1,'repeated': 2,'clusters_per_class': 2}]

    for samples_amount, features_amount, classes_amount, features_options in zip(samples, features, classes, options):
        print(f'\n\n\n Количество объектов {samples_amount}, '
              f'количество признаков {features_amount}, '
              f'количество классов {classes_amount}, '
              f'дополнительные опции {features_options}')

        x_data_train, y_data_train, x_data_test, y_data_test = prepare_classification_dataset(samples_amount,
                                                                                              features_amount,
                                                                                              classes_amount,
                                                                                              features_options)

        for preproc_func in [EmptyStrategy, Scaling, Normalization, ImputationStrategy]:
            print(str(preproc_func))
            if single == True:
                print('Single')
                chain = Chain(PrimaryNode(model, manual_preprocessing_func=preproc_func))
            else:
                print('Multi')
                node_1lv_1 = PrimaryNode('svc',
                                         manual_preprocessing_func=preproc_func)
                node_1lv_2 = PrimaryNode('logit',
                                         manual_preprocessing_func=preproc_func)

                node_2lv_1 = SecondaryNode('rf',
                                           manual_preprocessing_func=preproc_func,
                                           nodes_from=[node_1lv_1])
                node_2lv_2 = SecondaryNode('xgboost',
                                           manual_preprocessing_func=preproc_func,
                                           nodes_from=[node_1lv_2])

                # Root node - make final prediction
                node_final = SecondaryNode(model,
                                           manual_preprocessing_func=preproc_func,
                                           nodes_from=[node_2lv_1,
                                                       node_2lv_2])
                chain = Chain(node_final)


            task = Task(TaskTypesEnum.classification)

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

            chain.fit(train_input, verbose=True)

            # Predict
            predicted_labels = chain.predict(predict_input)
            if any(model == acceptable_model for acceptable_model in ['logit', 'lda', 'qda', 'mlp', 'svc',
                                                                      'xgboost', 'bernb']):
                preds = np.array(predicted_labels.predict)
                preds[preds >= 0.5] = 1
                preds[preds < 0.5] = 0
            else:
                preds = np.array(predicted_labels.predict, dtype=int)

            print(f'Предсказанные метки: {preds[:5]}')
            y_data_test = np.ravel(y_data_test)
            print(f'Действительные метки: {y_data_test[:5]}')
            print(f"{f1_score(y_data_test, preds, average='macro'):.2f}\n")
            chain = None
