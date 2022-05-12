from sklearn.metrics import roc_auc_score as roc_auc

from cases.data.data_utils import get_scoring_case_data_paths
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.utilities.project_import_export import export_project_to_zip, import_project_from_zip

if __name__ == '__main__':
    train_file_path, test_file_path = get_scoring_case_data_paths()
    train_data = InputData.from_csv(train_file_path)
    test_data = InputData.from_csv(test_file_path)

    pipeline = Pipeline(PrimaryNode('rf'))

    # Export project to zipfile to directory
    export_project_to_zip(zip_name='example', opt_history=None,
                          pipeline=pipeline, train_data=train_data, test_data=test_data)

    # Import project from zipfile to pipeline and InputData objects.
    pipeline, train_data, test_data, opt_history = \
        import_project_from_zip('example')

    pipeline.fit(train_data)
    prediction = pipeline.predict(test_data)

    print(roc_auc(test_data.target, prediction.predict))
