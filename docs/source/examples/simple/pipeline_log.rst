.. _log_example_page:

Logging Example
===============

This example demonstrates how to integrate logging into a machine learning pipeline using the `golem` framework. The main task solved here is the creation and fitting of a complex classification pipeline, with detailed logging of the process.

Overview
--------

The example showcases the use of logging to track the execution of a machine learning pipeline. It includes steps to set up a log file, create a classification pipeline, and fit the pipeline to training data. The logging mechanism is used to record important events and status updates during the execution of the pipeline.

Step-by-Step Guide
------------------

1. **Import Necessary Modules**

   The first block of code imports the necessary modules and functions required for the example.

   .. code-block:: python

      import logging
      import pathlib

      from golem.core.log import Log

      from examples.simple.classification.classification_pipelines import classification_complex_pipeline
      from examples.simple.pipeline_tune import get_case_train_test_data

2. **Define the `run_log_example` Function**

   This function takes a `log_file` parameter and performs the following tasks:

   a. **Fetch Training Data**

      The function starts by fetching training data using the `get_case_train_test_data` function.

      .. code-block:: python

         def run_log_example(log_file):
             train_data, _ = get_case_train_test_data()

   b. **Initialize Logging**

      It then initializes a logger with the specified log file and sets the output logging level to `logging.FATAL`. The logger is configured with a prefix derived from the stem of the current file's path.

      .. code-block:: python

             log = Log(log_file=log_file, output_logging_level=logging.FATAL).get_adapter(prefix=pathlib.Path(__file__).stem)

   c. **Create and Fit the Classification Pipeline**

      The function logs the start of creating the pipeline, creates the pipeline using `classification_complex_pipeline`, and then logs the start of fitting the pipeline. The pipeline is fitted to the training data.

      .. code-block:: python

             log.info('start creating pipeline')
             pipeline = classification_complex_pipeline()

             log.info('start fitting pipeline')
             pipeline.fit(train_data)

3. **Run the Example**

   The example is executed by calling the `run_log_example` function with a specified log file.

   .. code-block:: python

      if __name__ == '__main__':
          run_log_example(log_file='example_log.log')

Usage
-----

To use this example, you can copy and paste the provided code into your Python environment. Ensure that the necessary modules are available in your environment. You can modify the `log_file` parameter to specify a different log file or adjust the logging level as needed.

.. note::
   This example assumes the availability of certain functions and modules within the `golem` framework. Ensure that these are correctly imported and available in your environment.

.. seealso::
   For more detailed information on logging and the `golem` framework, refer to the official documentation.

.. _golem_framework: https://golem-framework.readthedocs.io/

This documentation page provides a comprehensive guide to understanding and using the logging example within the `golem` framework. It ensures that users can easily follow the steps and adapt the code to their own purposes.