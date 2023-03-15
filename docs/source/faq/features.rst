Features
==========

How does FEDOT avoid data leakage and bias?
-------------------------------------------

    Before the usage of AutoML, the dataset is usually splitting to training and test part.
    Then, the cross-validation is used to estimate the value of fitness functions. The default number of folds is 5.
    It can be specified manually using  ``cv_folds`` parameter.

    To deal with potential bias, additional metrics can be passed to optimiser using ``metric`` parameter.

Which algorithms are available as building block for optimiser?
---------------------------------------------------------------

    The names and metadata of available ML models can be obtained from
    `JSON <https://github.com/aimclub/FEDOT/blob/master/fedot/core/repository/data/model_repository.json>`__ file,
    as well as `data operations <https://github.com/nccr-itmo/FEDOT/blob/master/fedot/core/repository/data/data_operation_repository.json>`__ file.

What hyperparameter options were available to each algorithm?
-------------------------------------------------------------

    The ranges for tuning of hyperparameters are presented in
    `search_space.py <https://github.com/aimclub/FEDOT/blob/master/fedot/core/pipelines/tuning/search_space.py>`__ file.
    The default values of hyperparameters are available in
    `JSON <https://github.com/aimclub/FEDOT/blob/master/fedot/core/repository/data/default_operation_params.json>`__ file.