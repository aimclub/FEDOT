
.. _image_classification_example:

Image Classification Example
============================

This example demonstrates how to use the Fedot framework to solve an image classification problem using a Convolutional Neural Network (CNN) pipeline. The example uses the MNIST dataset, which is a set of 70,000 small images of digits handwritten by high school students and employees of the US Census Bureau.

Overview
--------

The example is structured into several logical blocks:

1. **Module Import and Requirement Check**: The code starts by importing necessary modules and checking if TensorFlow is installed. If not, it warns the user to install it.
2. **Metric Calculation Function**: A function to calculate the ROC AUC score for validation.
3. **Main Function**: A function to run the image classification problem, which includes setting up the task, preparing the datasets, creating and fitting the pipeline, and evaluating the model.
4. **Main Execution Block**: The main block where the training and testing datasets are loaded, and the main function is called.

Step-by-Step Guide
------------------

1. **Module Import and Requirement Check**

   .. code-block:: python

      from golem.utilities.requirements_notificator import warn_requirement

      try:
          import tensorflow as tf
      except ModuleNotFoundError:
          warn_requirement('tensorflow', 'fedot[extra]')

      from sklearn.metrics import roc_auc_score as roc_auc

      from examples.simple.classification.classification_pipelines import cnn_composite_pipeline
      from fedot.core.data.data import InputData, OutputData
      from fedot.core.repository.tasks import Task, TaskTypesEnum
      from fedot.core.utils import set_random_seed

   This block imports the necessary modules and checks for the presence of TensorFlow. If TensorFlow is not found, it notifies the user to install it.

2. **Metric Calculation Function**

   .. code-block:: python

      def calculate_validation_metric(predicted: OutputData, dataset_to_validate: InputData) -> float:
          # the quality assessment for the simulation results
          roc_auc_value = roc_auc(y_true=dataset_to_validate.target,
                                  y_score=predicted.predict,
                                  multi_class="ovo")
          return roc_auc_value

   This function calculates the ROC AUC score for the validation dataset. It takes the predicted output and the validation dataset as inputs and returns the ROC AUC value.

3. **Main Function**

   .. code-block:: python

      def run_image_classification_problem(train_dataset: tuple,
                                           test_dataset: tuple,
                                           composite_flag: bool = True):
          task = Task(TaskTypesEnum.classification)

          x_train, y_train = train_dataset[0], train_dataset[1]
          x_test, y_test = test_dataset[0], test_dataset[1]

          dataset_to_train = InputData.from_image(images=x_train,
                                                  labels=y_train,
                                                  task=task)
          dataset_to_validate = InputData.from_image(images=x_test,
                                                     labels=y_test,
                                                     task=task)

          pipeline = cnn_composite_pipeline(composite_flag)
          pipeline.fit(input_data=dataset_to_train)
          predictions = pipeline.predict(dataset_to_validate)
          roc_auc_on_valid = calculate_validation_metric(predictions,
                                                         dataset_to_validate)
          return roc_auc_on_valid, dataset_to_train, dataset_to_validate

   This function sets up the classification task, prepares the training and validation datasets, creates a CNN pipeline, fits the model, makes predictions, and calculates the ROC AUC score on the validation set.

4. **Main Execution Block**

   .. code-block:: python

      if __name__ == '__main__':
          set_random_seed(1)

          training_set, testing_set = tf.keras.datasets.mnist.load_data(path='mnist.npz')
          roc_auc_on_valid, dataset_to_train, dataset_to_validate = run_image_classification_problem(
              train_dataset=training_set,
              test_dataset=testing_set)

   In this block, the random seed is set, the MNIST dataset is loaded, and the main function is called with the training and testing datasets.

Usage
-----

To use this example, you can copy and paste the provided code into your Python environment. Ensure that you have the required dependencies installed, such as TensorFlow and Fedot with the 'extra' package. You can then run the script to see the ROC AUC score for the image classification task on the MNIST dataset.

.. note::
   Make sure to have the necessary permissions and paths set correctly to load the MNIST dataset.