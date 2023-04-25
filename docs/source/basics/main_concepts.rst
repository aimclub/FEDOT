Main Concepts
=============

The main framework concepts are as follows:

- **Flexibility.** FEDOT can be used to automate the construction of solutions for various `problems <https://fedot.readthedocs.io/en/master/introduction/fedot_features/main_features.html#involved-tasks>`_, `data types <https://fedot.readthedocs.io/en/master/introduction/fedot_features/automation_features.html#data-nature>`_ (texts, images, tables), and :doc:`models </advanced/automated_pipelines_design>`;
- **Extensibility.** Pipeline optimization algorithms are data- and task-independent, yet you can use :doc:`special strategies </api/strategies>` for specific tasks or data types (time-series forecasting, NLP, tabular data, etc.) to increase the efficiency;
- **Integrability.** FEDOT supports widely used ML libraries (Scikit-learn, CatBoost, XGBoost, etc.) and allows you to integrate `custom ones <https://fedot.readthedocs.io/en/master/api/strategies#module-fedot.core.operations.evaluation.custom>`_;
- **Tuningability.** Various :doc:`hyper-parameters tuning methods </advanced/hyperparameters_tuning>` are supported including models' custom evaluation metrics and search spaces;
- **Versatility.** FEDOT is :doc:`not limited to specific modeling tasks </advanced/architecture>`, for example, it can be used in ODE or PDE;
- **Reproducibility.** Resulting pipelines can be :doc:`exported separately as JSON </advanced/pipeline_import_export>` or :doc:`together with your input data as ZIP archive </advanced/project_import_export>` for experiments reproducibility;
- **Customizability.** FEDOT allows `managing models complexity <https://fedot.readthedocs.io/en/master/introduction/fedot_features/automation_features.html#models-used>`_ and thereby achieving desired quality.
