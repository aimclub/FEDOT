import os

import h2o
from h2o.automl import H2OAutoML

from benchmark.benchmark_utils import get_initial_data_paths


def run_h2o():
    train_data_path, test_data_path = get_initial_data_paths()

    h2o.init(ip="localhost", port=8888)

    # Data preprocessing
    target_column_name = 'default'
    train_frame = h2o.import_file(train_data_path)
    test_frame = h2o.import_file(test_data_path)
    predictors = train_frame.columns.remove(target_column_name)
    train_frame[target_column_name] = train_frame[target_column_name].asfactor()
    test_frame[target_column_name] = test_frame[target_column_name].asfactor()

    result_filename = 'last_saved_model'
    current_path = str(os.path.dirname(__file__))
    exported_model_path = os.path.join(current_path, result_filename)

    if result_filename not in os.listdir('.'):
        model = H2OAutoML(max_models=20, seed=1)
        model.train(x=predictors, y=target_column_name, training_frame=train_frame, validation_frame=test_frame)
        best_model = model.leader
        temp_exported_model_path = h2o.save_model(model=best_model, path='.')

        os.renames(temp_exported_model_path, exported_model_path)

    imported_model = h2o.load_model(exported_model_path)

    train_roc_auc_value = imported_model.auc(train=True)
    test_roc_auc_value = imported_model.auc(valid=True)
    model_performance_table = imported_model.model_performance(test_frame)

    print(f'TOTAL_PERFORMANCE_METRICS: {model_performance_table}')
    print(f'ROC_AUC_train: {train_roc_auc_value}')
    print(f'ROC_AUC_test: {test_roc_auc_value}')
    imported_model.varimp_plot()


if __name__ == '__main__':
    run_h2o()
