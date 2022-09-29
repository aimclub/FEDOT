How to save fitted models and interpreting the pipeline in JSON format
======================================================================

FEDOT works mainly with the *'Pipeline'* object, which is a pipeline of models. For more
convenient use of the framework, we provide the ability
to upload and download pipelines, and their models for further editing, visual
representation, or data transfer. Here are some simple steps to export
and import pipeline structure.

.. .. figure::  img_utilities/pipeline_json.png
..    :align:   center

Exporting a model pipeline
--------------------------

The Pipeline object has a *'save_pipeline'* method that takes a single argument,
the path to where the JSON object and fitted models will be saved.
You can specify the path to save files with the folder name:

- ~/project/model/my_pipeline,

this way your pipeline and trained models will be saved in a folder in the following hierarchy:

- ~/project/model/my_pipeline:
    - my_pipeline.json
    - fitted_models:
        - model_0.pkl
        - model_2.pkl
        - ...

**Example of use:**

.. code-block:: python

    from cases.data.data_utils import get_scoring_case_data_paths
    from fedot.core.pipelines.pipeline import Pipeline
    from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
    from fedot.core.data.data import InputData

    train_file_path, test_file_path = get_scoring_case_data_paths()
    train_data = InputData.from_csv(train_file_path)

    pipeline = Pipeline()
    node_logit = PrimaryNode('logit')

    node_lda = PrimaryNode('lda')
    node_lda.custom_params = {'n_components': 1}

    node_xgboost = PrimaryNode('xgboost')

    node_knn_second = SecondaryNode('knn')
    node_knn_second.custom_params = {'n_neighbors': 5}
    node_knn_second.nodes_from = [node_logit, node_lda, node_xgboost]

    pipeline.add_node(node_knn_second)
    pipeline.fit(train_data)

    pipeline.save_pipeline("data/my_pipeline")

The *'save_pipeline'* method:

1. saves the pipeline's fitted models to path ``test/data/my_pipeline/fitted_models``,
2. saves JSON object in the file ``test/data/my_pipeline/my_pipeline.json``,
3. returns a JSON-like-object

.. code-block:: json

    {
        "total_pipeline_models": {
            "logit": 1,
            "lda": 1,
            "xgboost": 1,
            "knn": 1
        },
        "depth": 2,
        "nodes": [
            {
                "model_id": 1,
                "model_type": "logit",
                "model_name": "LogisticRegression",
                "custom_params": "default_params",
                "params": {
                    
                },
                "nodes_from": [],
                "fitted_model_path": "fitted_models/model_1.pkl",
                "preprocessor": "scaling_with_imputation"
            },
            {
                "model_id": 2,
                "model_type": "lda",
                "model_name": "LinearDiscriminantAnalysis",
                "custom_params": {
                    "n_components": 1
                },
                "params": {
                    
                },
                "nodes_from": [],
                "fitted_model_path": "fitted_models/model_2.pkl",
                "preprocessor": "scaling_with_imputation"
            },
            {
                "model_id": 3,
                "model_type": "xgboost",
                "model_name": "XGBClassifier",
                "custom_params": "default_params",
                "params": {
                    
                },
                "nodes_from": [],
                "fitted_model_path": "fitted_models/model_3.pkl",
                "preprocessor": "scaling_with_imputation"
            },
            {
                "model_id": 0,
                "model_type": "knn",
                "model_name": "KNeighborsClassifier",
                "custom_params": {
                    "n_neighbors": 5
                },
                "params": {
                    
                },
                "nodes_from": [
                    1,
                    2,
                    3
                ],
                "fitted_model_path": "fitted_models/model_0.pkl",
                "preprocessor": "scaling_with_imputation"
            }
        ]
    }


**NOTE:** *'params'* are all parameters consisting of:

- parameters for tuning (custom_params),
- standard model parameters in the framework

Model Pipeline import
---------------------

To import a pipeline, you need to create an empty *'Pipeline'* object, or an
already used one, but all data will be overwritten during import. The
*'load_pipeline'* method takes the path to a file with the JSON extension
as an argument.

**Example of using a model:**

.. code-block:: python

    from sklearn.metrics import mean_squared_error

    test_data = InputData.from_csv(test_file_path)

    pipeline = Pipeline()
    pipeline.load_pipeline("data/Month:Day:Year, Time Period my_pipeline/my_pipeline.json")
    predicted_values = pipeline.predict(test_data).predict
    actual_values = test_data.target

    mean_squared_error(predicted_values, actual_values)

**NOTE:** Required fields for loading the model are: **'model_id'**, **'model_type'**, **'preprocessor'**,
**'params'**, **'nodes_from'**. The consequence is that you can
create an unusual pipeline.

Now you can upload models, share them, and edit them in a convenient JSON format.
