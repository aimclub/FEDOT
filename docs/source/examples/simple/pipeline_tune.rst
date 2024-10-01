
.. _classification_pipeline_tuning_example:

Classification Pipeline Tuning Example
==================================================================

This example demonstrates how to tune a classification pipeline using the SimultaneousTuner from the Fedot framework. The goal is to improve the ROC AUC score of a classification model by iteratively tuning the pipeline and evaluating its performance on a test dataset.

Overview
--------

The example is structured into several logical blocks:

1. **Data Loading**: The `get_case_train_test_data` function loads training and testing data from CSV files.
2. **Pipeline Initialization**: A classification pipeline is initialized using the `classification_complex_pipeline` function.
3. **Initial Prediction**: The pipeline is fitted on the training data and predictions are made on the test data to obtain an initial ROC AUC score.
4. **Pipeline Tuning**: The `pipeline_tuning` function tunes the pipeline using the SimultaneousTuner and evaluates its performance over multiple iterations.
5. **Results Analysis**: The final ROC AUC scores before and after tuning are compared and displayed.

Step-by-Step Guide
------------------

### Data Loading

.. code-block:: python

    def get_case_train_test_data():
        """ Function for getting data for train and validation """
        train_file_path, test_file_path = get_scoring_case_data_paths()

        train_data = InputData.from_csv(train_file_path)
        test_data = InputData.from_csv(test_file_path)
        return train_data, test_data

### Pipeline Initialization

.. code-block:: python

    # Pipeline composition
    pipeline = classification_complex_pipeline()

### Initial Prediction

.. code-block:: python

    # Before tuning prediction
    pipeline.fit(train_data)
    before_tuning_predicted = pipeline.predict(test_data)
    bfr_tun_roc_auc = roc_auc(y_true=test_data.target, y_score=before_tuning_predicted.predict)

### Pipeline Tuning

.. code-block:: python

    def pipeline_tuning(pipeline: Pipeline, train_data: InputData, test_data: InputData, local_iter: int, tuner_iter_num: int = 30) -> (float, list):
        """ Function for tuning pipeline with SimultaneousTuner

        :param pipeline: pipeline to tune
        :param train_data: InputData for train
        :param test_data: InputData for validation
        :param local_iter: amount of tuner launches
        :param tuner_iter_num: amount of iterations, which tuner will perform

        :return mean_metric: mean value of ROC AUC metric
        :return several_iter_scores_test: list with metrics
        """
        several_iter_scores_test = []
        tuner = TunerBuilder(train_data.task) \
            .with_tuner(SimultaneousTuner) \
            .with_metric(ClassificationMetricsEnum.ROCAUC) \
            .with_iterations(tuner_iter_num) \
            .build(train_data)
        for iteration in range(local_iter):
            print(f'current local iteration {iteration}')

            # Pipeline tuning
            tuned_pipeline = tuner.tune(pipeline)

            # After tuning prediction
            tuned_pipeline.fit(train_data)
            after_tuning_predicted = tuned_pipeline.predict(test_data)

            # Metrics
            aft_tun_roc_auc = roc_auc(y_true=test_data.target, y_score=after_tuning_predicted.predict)
            several_iter_scores_test.append(aft_tun_roc_auc)

        max_metric = float(np.max(several_iter_scores_test))
        return max_metric, several_iter_scores_test

### Results Analysis

.. code-block:: python

    if __name__ == '__main__':
        train_data, test_data = get_case_train_test_data()

        # Pipeline composition
        pipeline = classification_complex_pipeline()

        # Before tuning prediction
        pipeline.fit(train_data)
        before_tuning_predicted = pipeline.predict(test_data)
        bfr_tun_roc_auc = roc_auc(y_true=test_data.target, y_score=before_tuning_predicted.predict)

        local_iter = 5
        # Pipeline tuning
        after_tune_roc_auc, several_iter_scores_test = pipeline_tuning(pipeline=pipeline, train_data=train_data, test_data=test_data, local_iter=local_iter)

        print(f'Several test scores {several_iter_scores_test}')
        print(f'Maximal test score over {local_iter} iterations: {after_tune_roc_auc}')
        print(f'ROC-AUC before tuning {round(bfr_tun_roc_auc, 3)}')
        print(f'ROC-AUC after tuning {round(after_tune_roc_auc, 3)}')

This documentation page provides a comprehensive understanding of the classification pipeline tuning example. Users can copy and paste the provided code snippets to reproduce the example and adapt it to their own classification tasks.

This .rst formatted documentation page is structured to guide the user through the example, explaining each logical block and providing the full code for reference. The user should be able to understand the example and apply it to their own purposes.