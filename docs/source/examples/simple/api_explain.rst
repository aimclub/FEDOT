
Fedot API Explain Example
=========================

Overview
--------

This example demonstrates how to use the Fedot framework to build a classification model and explain its predictions using a surrogate decision tree. The example uses a dataset from the Fedot project's cases directory, specifically the cancer training dataset. The model is built using a predefined Random Forest model and then explained using a surrogate decision tree method.

Step-by-Step Guide
------------------

1. **Import Necessary Libraries**

   The first step is to import the necessary libraries, which include `pandas` for data manipulation and `Fedot` for the machine learning framework.

   .. code-block:: python

      import pandas as pd
      from fedot import Fedot
      from fedot.core.utils import fedot_project_root

2. **Define the Function to Run the Example**

   The function `run_api_explain_example` is defined with parameters for visualization, timeout, and whether to perform model tuning.

   .. code-block:: python

      def run_api_explain_example(visualization=False, timeout=None, with_tuning=True):

3. **Load the Training Data**

   The training data is loaded from a CSV file located in the Fedot project's cases/data/cancer directory.

   .. code-block:: python

        train_data = pd.read_csv(f'{fedot_project_root()}/cases/data/cancer/cancer_train.csv', index_col=0)

4. **Prepare Visualization Parameters**

   The feature and class names are extracted for visualization purposes.

   .. code-block:: python

        feature_names = train_data.columns.tolist()
        target_name = feature_names.pop()
        target = train_data[target_name]
        class_names = target.unique().astype(str).tolist()

5. **Build the Classification Model**

   A Fedot model is initialized with the problem type 'classification', a timeout, and whether to perform model tuning. The model is then fitted using the training data and a predefined Random Forest model.

   .. code-block:: python

        model = Fedot(problem='classification', timeout=timeout, with_tuning=with_tuning)
        model.fit(features=train_data, target=target_name, predefined_model='rf')

6. **Explain the Model Predictions**

   The model's predictions are explained using a surrogate decision tree method. If visualization is enabled, the explanation is saved as an image.

   .. code-block:: python

        explainer = model.explain(
            method='surrogate_dt', visualization=visualization,
            save_path=figure_path, dpi=200, feature_names=feature_names,
            class_names=class_names, precision=6
        )

7. **Run the Example**

   The example is executed with visualization enabled and a timeout of 5 seconds.

   .. code-block:: python

        if __name__ == '__main__':
            run_api_explain_example(visualization=True, timeout=5)

Conclusion
----------

This example showcases the use of the Fedot framework for building a classification model and explaining its predictions. It demonstrates how to load data, build a model, and use a surrogate decision tree to explain the model's decisions. The example can be easily adapted for different datasets and models by modifying the data loading and model configuration sections.
