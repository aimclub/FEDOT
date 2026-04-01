# benchmark/evaluation/metrics.py
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


class ForecastEvaluator:
    """Класс для оценки качества прогнозирования"""

    def __init__(self):
        self.metrics_functions = {
            'MASE': self._calculate_mase,
            'sMAPE': self._calculate_smape,
            'RMSE': self._calculate_rmse,
            'MAE': self._calculate_mae,
            'MAPE': self._calculate_mape,
            'OWA': self._calculate_owa  # Overall Weighted Average для M4
        }

    def evaluate_forecast(self, y_true: np.ndarray, y_pred: np.ndarray,
                          y_train: np.ndarray, metrics: List[str]) -> Dict[str, float]:
        """Оценка прогноза по нескольким метрикам"""
        results = {}

        for metric in metrics:
            if metric in self.metrics_functions:
                try:
                    results[metric] = self.metrics_functions[metric](y_true, y_pred, y_train)
                except Exception as e:
                    results[metric] = np.nan
                    print(f"Error calculating {metric}: {e}")

        return results

    def _calculate_mase(self, y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
        """Mean Absolute Scaled Error"""
        mae = np.mean(np.abs(y_true - y_pred))

        # Naive forecast errors
        naive_errors = np.mean(np.abs(np.diff(y_train)))

        if naive_errors == 0:
            return np.inf if mae > 0 else 0

        return mae / naive_errors

    def _calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """Symmetric Mean Absolute Percentage Error"""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        denominator = np.where(denominator == 0, 1e-10, denominator)  # Avoid division by zero
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / denominator)

    def _calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """Root Mean Square Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def _calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)

    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """Mean Absolute Percentage Error"""
        denominator = np.where(y_true == 0, 1e-10, y_true)  # Avoid division by zero
        return 100 * np.mean(np.abs((y_true - y_pred) / denominator))

    def _calculate_owa(self, y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
        """Overall Weighted Average (M4 competition metric)"""
        # OWA = 0.5 * (sMAPE/sMAPE_naive2 + MASE/MASE_naive2)
        smape = self._calculate_smape(y_true, y_pred)
        mase = self._calculate_mase(y_true, y_pred, y_train)

        # Здесь должны быть benchmark значения для naive2
        # Временно используем упрощенную версию
        smape_naive2 = smape * 1.1  # Placeholder
        mase_naive2 = mase * 1.1  # Placeholder

        return 0.5 * (smape / smape_naive2 + mase / mase_naive2)


class ModelComparator:
    """Сравнение моделей и статистический анализ"""

    def __init__(self):
        self.evaluator = ForecastEvaluator()

    def compare_models(self, results_df: pd.DataFrame, metric: str = 'MASE') -> pd.DataFrame:
        """Сравнение моделей по выбранной метрике"""
        comparison = results_df.groupby('model')[metric].agg(['mean', 'std', 'count']).round(4)
        comparison['rank'] = comparison['mean'].rank()

        return comparison.sort_values('mean')

    def statistical_significance(self, results_df: pd.DataFrame, baseline_model: str,
                                 metric: str = 'MASE') -> pd.DataFrame:
        """Проверка статистической значимости различий"""
        from scipy import stats

        models = results_df['model'].unique()
        significance_matrix = pd.DataFrame(index=models, columns=models)

        for model1 in models:
            for model2 in models:
                if model1 == model2:
                    significance_matrix.loc[model1, model2] = '-'
                else:
                    values1 = results_df[results_df['model'] == model1][metric].values
                    values2 = results_df[results_df['model'] == model2][metric].values

                    if len(values1) > 1 and len(values2) > 1:
                        # t-test для парных сравнений
                        t_stat, p_value = stats.ttest_rel(values1, values2)
                        significance_matrix.loc[model1, model2] = f"p={p_value:.4f}"
                    else:
                        significance_matrix.loc[model1, model2] = 'N/A'

        return significance_matrix
