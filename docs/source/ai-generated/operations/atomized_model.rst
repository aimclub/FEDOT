Atomized Model
==============

This module defines a class for AtomizedModel objects.

:class:`AtomizedModel`:
-----------------------

Class which replaces Operation class for AtomizedModel object.

Parameters:
    - pipeline: Pipeline for the AtomizedModel.

Methods:
    - fit(self, params, data): Fits the AtomizedModel.
    - predict(self, fitted_operation, data, params, output_mode): Predicts using the AtomizedModel.
    - predict_for_fit(self, fitted_operation, data, params, output_mode): Predicts for the fit stage.
    - fine_tune(self, metric_function, input_data, iterations, timeout): Fine-tunes hyperparameters.
    - metadata(self): Provides metadata for the AtomizedModel.
    - description(self, operation_params): Provides a description for the AtomizedModel.
    - assign_tabular_column_types(output_data, output_mode): Assigns column types for tabular data.

