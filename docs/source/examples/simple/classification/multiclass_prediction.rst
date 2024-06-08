
.. _multi_clf_examples_from_excel:

=================================================================================
Multi-Class Classification Examples from Excel Files
=================================================================================

This example demonstrates how to use the `FEDOT <https://github.com/nccr-itmo/FEDOT>`_ framework to perform multi-class classification tasks using data stored in Excel files. The example covers the entire process from reading the data, training a model, validating its performance, and applying it to new data.

Prerequisites
-------------

Ensure you have the following Python packages installed:

- `pandas`
- `openpyxl`
- `FEDOT`
- `sklearn`

You can install the required packages using pip:

.. code-block:: bash

    pip install pandas openpyxl fedot[examples] sklearn

Example Overview
-----------------

The example is structured into several functions that handle different parts of the machine learning pipeline:

1. **Data Loading and Preprocessing**: The `create_multi_clf_examples_from_excel` function reads an Excel file, splits the data into training and testing sets, and optionally saves the data to CSV files.

2. **Model Training**: The `get_model` function trains a model using the training data. It uses a genetic programming-based composer to find the optimal model structure.

3. **Model Application**: The `apply_model_to_data` function applies the trained model to new data and generates predictions.

4. **Model Validation**: The `validate_model_quality` function evaluates the model's performance using the ROC AUC metric.

Step-by-Step Guide
------------------

1. **Data Loading and Preprocessing**

   The `create_multi_clf_examples_from_excel` function is responsible for loading data from an Excel file and preparing it for model training. Here's how it works:

   .. code-block:: python

    def create_multi_clf_examples_from_excel(file_path: str, return_df: bool = False):
        df = pd.read_excel(file_path, engine='openpyxl')
        train, test = split_data(df)
        file_dir_name = file_path.replace('.', '/').split('/')[-2]
        file_csv_name = f'{file_dir_name}.csv'
        directory_names = ['examples', 'data', file_dir_name]

        ensure_directory_exists(directory_names)
        if return_df:
            path = os.path.join(directory_names[0], directory_names[1], directory_names[2], file_csv_name)
            full_file_path = os.path.join(str(fedot_project_root()), path)
            save_file_to_csv(df, full_file_path)
            return df, full_file_path
        else:
            full_train_file_path, full_test_file_path = get_split_data_paths(directory_names)
            save_file_to_csv(train, full_train_file_path)
            save_file_to_csv(train, full_test_file_path)
            return full_train_file_path, full_test_file_path

2. **Model Training**

   The `get_model` function trains a model using the training data. It uses a genetic programming-based composer to find the optimal model structure.

   .. code-block:: python

    def get_model(train_file_path: str, cur_lead_time: datetime.timedelta = timedelta(seconds=60)):
        task = Task(task_type=TaskTypesEnum.classification)
        dataset_to_compose = InputData.from_csv(train_file_path, task=task)

        models_repo = OperationTypesRepository()
        available_model_types = models_repo.suitable_operation(task_type=task.task_type, tags=['simple'])

        metric_function = ClassificationMetricsEnum.ROCAUC_penalty

        composer_requirements = PipelineComposerRequirements(
            primary=available_model_types, secondary=available_model_types,
            timeout=cur_lead_time)

        builder = ComposerBuilder(task).with_requirements(composer_requirements).with_metrics(metric_function)
        composer = builder.build()

        pipeline_evo_composed = composer.compose_pipeline(data=dataset_to_compose)
        pipeline_evo_composed.fit(input_data=dataset_to_compose)

        return pipeline_evo_composed

3. **Model Application**

   The `apply_model_to_data` function applies the trained model to new data and generates predictions.

   .. code-block:: python

    def apply_model_to_data(model: Pipeline, data_path: str):
        df, file_path = create_multi_clf_examples_from_excel(data_path, return_df=True)
        dataset_to_apply = InputData.from_csv(file_path, target_columns=None)
        evo_predicted = model.predict(dataset_to_apply)
        df['forecast'] = probs_to_labels(evo_predicted.predict)
        return df

4. **Model Validation**

   The `validate_model_quality` function evaluates the model's performance using the ROC AUC metric.

   .. code-block:: python

    def validate_model_quality(model: Pipeline, data_path: str):
        dataset_to_validate = InputData.from_csv(data_path)
        predicted_labels = model.predict(dataset_to_validate).predict

        roc_auc_valid = round(roc_auc(y_true=test_data.target,
                                      y_score=predicted_labels,
                                      multi_class='ovo',
                                      average='macro'), 3)
        return roc_auc_valid

Running the Example
-------------------

To run the example, execute the following code:

.. code-block:: python

    if __name__ == '__main__':
        set_random_seed(1)

        data_path = Path('../../data')
        file_path_first = data_path.joinpath('example1.xlsx')
        file_path_second = data_path.joinpath('example2.xlsx')
        file_path_third = data_path.joinpath('example3.xlsx')

        train_file_path, test_file_path = create_multi_clf_examples_from_excel(file_path_first)
        test_data = InputData.from_csv(test_file_path)

        fitted_model = get_model(train_file_path)

        fitted_model.show()

        roc_auc_score = validate_model_quality(fitted_model, test_file_path)
        print(f'ROC AUC metric is {roc_auc_score}')

        final_prediction_first = apply_model_to_data(fitted_model, file_path_second)
        print(final_prediction_first['forecast'])

        final_prediction_second = apply_model_to_data(fitted_model, file_path_third)
        print(final_prediction_second['forecast'])

This will load data from three Excel files, train a model, validate its performance, and apply it to new data, printing the ROC AUC score and the model's predictions.

This documentation page provides a comprehensive guide to the example code, ensuring that users can understand and replicate the process for their own purposes.