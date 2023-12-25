import os

from examples.advanced.customization.image_classification_with_custom_models import run_image_classification_automl


def test_image_classification_automl():
    test_data_path = '../../data/test_data.npy'
    test_labels_path = '../../data/test_labels.npy'
    train_data_path = '../../data/training_data.npy'
    train_labels_path = '../../data/training_labels.npy'

    test_file_path = str(os.path.dirname(__file__))
    training_path_features = os.path.join(test_file_path, train_data_path)
    training_path_labels = os.path.join(test_file_path, train_labels_path)
    test_path_features = os.path.join(test_file_path, test_data_path)
    test_path_labels = os.path.join(test_file_path, test_labels_path)

    roc_auc_on_valid, dataset_to_train, dataset_to_validate = run_image_classification_automl(
        train_dataset=(training_path_features,
                       training_path_labels),
        test_dataset=(test_path_features,
                      test_path_labels))

    return roc_auc_on_valid, dataset_to_train, dataset_to_validate
