How to save fitted models and interpreting the chain in JSON format
===================================================================

FEDOT works mainly with the *'Chain'* object, which is a chain of models. For more
convenient use of the framework, we provide the ability
to upload and download chains, and their models for further editing, visual
representation, or data transfer. Here are some simple steps to export 
and import chain structure.

.. figure::  img/chain_to_json_main.png
   :align:   center

Exporting a model chain
-----------------------
      
The Chain object has a *'save_chain'* method that takes a single argument,
the path (relative or absolute) to which the JSON object will be saved. 
You can specify the path to save the JSON with the file name:

- home/user/project/model/my_chain.json,

or folder:

- home/user/project/model

in this case, the name of JSON file will be generated automatically:

- /home/user/project/model/unique_id.json

Also, the models that make up the chain will be saved to your home path
depending on your operating system:

- ~/Fedot/fitted_models

**Example of use:**

.. code-block:: python

    train_file_path, test_file_path = get_scoring_case_data_paths()
    train_data = InputData.from_csv(train_file_path)
    chain = Chain()
    node_logit = PrimaryNode('logit')
    
    node_lda = PrimaryNode('lda')
    node_lda.custom_params = {'n_components': 1}
    
    node_xgboost = PrimaryNode('xgboost')
    
    node_knn_second = SecondaryNode('knn')
    node_knn_second.custom_params = {'n_neighbors': 5}
    node_knn_second.nodes_from = [node_logit, node_lda, node_xgboost]
    
    chain.add_node(node_knn_second)
    chain.fit(train_data)
    
    chain.save_chain("test/data/my_chain.json")

The *'save_chain'* method:

1. saves the chain models to the home's storage,
2. saves JSON object in the file ``test/data/my_chain.json``,
3. returns a JSON-like-object

.. code-block:: json

    {
        "total_chain_types": {
            "knn": 1,
            "lda": 1,
            "logit": 1,
            "xgboost": 1
        },
        "depth": 2,
        "nodes": [
            {
                "model_id": "0",
                "model_type": "knn",
                "model_name": "KNeighborsClassifier",
                "custom_params": {
                    "n_neighbors": 5
                },
                "params": {
                    ...
                },
                "nodes_from": [
                    1,
                    2,
                    3
                ],
                "trained_model_path": "/home/user/.local/share/FEDOT/fitted_models/a3749e94-d387-4ff1-a7c5-ef232a384eec/model_0.pkl"
            },
            {
                "model_id": "1",
                "model_type": "logit",
                "model_name": "LogisticRegression",
                "custom_params": "default_params",
                "params": {
                    ...
                },
                "nodes_from": [],
                "trained_model_path": "/home/user/.local/share/FEDOT/fitted_models/a3749e94-d387-4ff1-a7c5-ef232a384eec/model_1.pkl"
            },
            {
                "model_id": "2",
                "model_type": "lda",
                "model_name": "LinearDiscriminantAnalysis",
                "custom_params": {
                    "n_components": 1
                },
                "params": {
                    ...
                },
                "nodes_from": [],
                "trained_model_path": "/home/user/.local/share/FEDOT/fitted_models/a3749e94-d387-4ff1-a7c5-ef232a384eec/model_2.pkl"
            },
            {
                "model_id": "3",
                "model_type": "xgboost",
                "model_name": "XGBClassifier",
                "custom_params": "default_params",
                "params": {
                    ...
                },
                "nodes_from": [],
                "trained_model_path": "/home/user/.local/share/FEDOT/fitted_models/a3749e94-d387-4ff1-a7c5-ef232a384eec/model_3.pkl"
            }
        ]
    }

Where *'params'* are all parameters consisting of:
- parameters for tuning (custom_params),
- standard model parameters in the framework

Model Chain import
------------------
      
To import a chain, you need to create an empty *'Chain'* object, or an
already used one, but all data will be overwritten during import. The
*'load_chain'* method takes the path to a file with the JSON extension
as an argument.

**Example of using a model with a minimal data set:**

.. code-block:: json

    "test/data/my_chain.json"

    {
      "nodes": [
            {
                "model_id": "0",
                "model_type": "knn",
                "params": {
                    "n_neighbors": 8
                },
                "nodes_from": [
                    1
                ]
            },
            {
                "model_id": "1",
                "model_type": "lda",
                "params": {
                    "n_components": 1
                },
                "nodes_from": []
            }
        ]
    }

.. code-block:: python

    chain = Chain()
    chain.load_chain("test/data/my_chain.json")

 
Required fields for loading the model are: *'model_id'*, *'model_type'*,
*'params' = {}*, *'nodes_from'= []*. The consequence is that you can
create an unusual chain.

Now you can upload models, share them, and edit them in a convenient JSON format.
 



