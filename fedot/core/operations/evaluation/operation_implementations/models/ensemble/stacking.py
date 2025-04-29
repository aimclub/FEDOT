from typing import Optional

import optuna
from sklearn.metrics import accuracy_score as accuracy, mean_squared_error as mse
from sklearn.linear_model import Ridge, Lasso, LogisticRegression

from golem.core.log import default_log

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.data.data_split import train_test_data_setup


class StackingImplementation(ModelImplementation):
    """Base class for stacking operations"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.log = default_log('Stacking')

        self.seed = 42
        self.max_iter = 50
        self.model = None
        self.metric = None

    def fit(self, input_data: InputData):
        pass

    def predict(self, input_data: InputData) -> OutputData:
        pass


class StackingClassifier(StackingImplementation):
    """Stacking implementation for classification tasks"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.metric = accuracy
        self.model = LogisticRegression

    def fit(self, input_data: InputData):
        """Fit the stacking model on input meta-data.

        Args:
            input_data: Input data containing predictions of previous models.
        """
        train, val = train_test_data_setup(input_data, split_ratio=0.9)
        penalties = ['l2', 'l1']
        model_score_array = []

        # Hyperparams tuning. Create pair (model: best_alpha)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner()
        )
        for pen in penalties:
            self.log.message(f"Starting optimization for "
                             f"linear regression with penalty: {pen}")

            def objective(trial):
                rev_alpha = trial.suggest_float("C", 0.001, 1000, log=True)

                est = self.model(
                    solver='saga',
                    penalty=pen,
                    C=rev_alpha,
                    random_state=self.seed
                )
                est.fit(X=train.features, y=train.target)
                pred_labels = est.predict(val.features)

                score = self.metric(val.target, pred_labels)
                return score

            study.optimize(objective, n_trials=self.max_iter)

            self.log.message(f"For penalty {pen} on validation set got "
                             f"{self.metric.__name__} = {round(study.best_value, 3)}")

            model = self.model(
                solver='saga',
                penalty=pen,
                C=study.best_params["C"],
                random_state=self.seed
            )
            model_score_array.append((model, study.best_value, pen))

        # Get best model by score and fit it
        best_model, best_score, penalty = max(model_score_array, key=lambda x: x[1])
        best_rev_alpha = round(study.best_params["C"], 6)
        self.model = best_model

        self.log.message(f"Chosen stacking model with "
                         f"penalty: {penalty} and 'alpha' parameter: {best_rev_alpha}")

        self.model.fit(input_data.features, input_data.target)
        return self

    def predict(self, input_data: InputData) -> OutputData:
        """Make labels predictions using the stacked model.

        Args:
            input_data: Input data containing predictions of previous models.
        """
        labels = self.model.predict(X=input_data.features)
        output_data = self._convert_to_output(input_data=input_data, predict=labels)
        return output_data

    def predict_proba(self, input_data: InputData) -> OutputData:
        """Make probabilities predictions using the stacked model.

        Args:
            input_data: Input data containing predictions of previous models.
        """
        probs = self.model.predict_proba(X=input_data.features)
        output_data = self._convert_to_output(input_data=input_data, predict=probs)
        return output_data


class StackingRegressor(StackingImplementation):
    """Stacking implementation for regression tasks"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.metric = mse
