model_params_with_bounds_by_model_name = {
    'xgboost': {
        'n_estimators': [10, 100],
        'max_depth': [1, 7],
        'learning_rate': [0.1, 0.9],
        'subsample': [0.05, 1.0],
        'min_child_weight': [1, 21]},
    'logit': {
        'C': [1e-2, 10.]},
    'knn': {
        'n_neighbors': [1, 50],
        'p': [1, 2]},
    'qda': {
        'reg_param': [0.1, 0.5]},
}

INTEGER_PARAMS = ['n_estimators', 'n_neighbors', 'p', 'min_child_weight', 'max_depth']
