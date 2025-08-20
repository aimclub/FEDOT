
Fedot Time Series Forecasting Example
================================================================

Overview
--------

This example demonstrates how to use the Fedot framework for time series forecasting. It sets up a Fedot model builder with specific configurations, applies it to multiple datasets, and evaluates the performance of the generated models.

Step-by-Step Guide
------------------

1. Importing Necessary Modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from fedot import FedotBuilder
    from fedot.core.utils import fedot_project_root

2. Setting Up the Fedot Builder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    if __name__ == '__main__':
        SEED = 42

        builder = (FedotBuilder('ts_forecasting')
                   .setup_composition(preset='fast_train', timeout=0.5, with_tuning=True, seed=SEED)
                   .setup_evolution(num_of_generations=3)
                   .setup_pipeline_evaluation(metric='mae'))

   In this block, the Fedot builder is initialized with a focus on time series forecasting. It sets up the composition with a fast training preset, a timeout of 0.5 seconds, tuning enabled, and a fixed seed for reproducibility. The evolution setup specifies 3 generations, and the evaluation metric is set to MAE (Mean Absolute Error).

3. Defining the Dataset Path
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    datasets_path = fedot_project_root() / 'examples/data/ts'

   This line defines the path to the directory containing the time series datasets.

4. Processing Each Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    resulting_models = {}
    for data_path in datasets_path.iterdir():
        if data_path.name == 'ts_sea_level.csv':
            continue
        fedot = builder.build()
        fedot.fit(data_path, target='value')
        fedot.predict(features=fedot.train_data, validation_blocks=2)
        fedot.plot_prediction()
        fedot.current_pipeline.show()
        resulting_models[data_path.stem] = fedot

   In this loop, each dataset file (excluding 'ts_sea_level.csv') is processed. For each file:

   - A Fedot model is built using the configured builder.
   - The model is trained on the dataset with the target column 'value'.
   - Predictions are made using the training data with 2 validation blocks.
   - A plot of the prediction is generated.
   - The current pipeline is displayed.
   - The model is stored in a dictionary with the dataset filename stem as the key.

Conclusion
----------

This example provides a comprehensive guide on using Fedot for time series forecasting. It demonstrates how to configure and use the Fedot builder to process multiple datasets, evaluate models, and visualize predictions. Users can easily adapt this example to their own datasets and forecasting tasks by modifying the dataset paths and configurations as needed.