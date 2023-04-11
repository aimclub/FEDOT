Main Concepts
=============

The main framework concepts are as follows:

- **Flexibility.** FEDOT can be used to automate the construction of solutions for various :doc:`problems </introduction/fedot_features/involved_tasks>`, :doc:`data types </introduction/fedot_features/automation/data_nature>` (texts, images, tables), and :doc:`models </advanced/automated_pipelines_design>`;
- **Extensibility.** Pipeline optimization algorithms are data- and task-independent, yet you can use :doc:`special strategies </api/strategies>` for specific tasks or data types (time-series forecasting, NLP, tabular data, etc.) to increase the efficiency;
- **Integrability.** FEDOT supports widely used ML libraries (Scikit-learn, CatBoost, XGBoost, etc.) and allows you to integrate `custom ones <https://fedot.readthedocs.io/en/master/api/strategies#module-fedot.core.operations.evaluation.custom>`_;
- **Tuningability.** Various :doc:`hyper-parameters tuning methods </advanced/hyperparameters_tuning>` are supported including models' custom evaluation metrics and search spaces;
- **Versatility.** FEDOT is :doc:`not limited to specific modeling tasks </advanced/architecture>`, for example, it can be used in ODE or PDE;
- **Reproducibility.** Resulting pipelines can be :doc:`exported separately as JSON </advanced/pipeline_import_export>` or :doc:`together with your input data as ZIP archive </advanced/project_import_export>` for experiments reproducibility;
- **Customizability.** FEDOT allows :doc:`managing models complexity </introduction/fedot_features/automation/models_used>` and thereby achieving desired quality.
