import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """Базовый интерфейс для всех моделей прогнозирования"""

    @abstractmethod
    def fit(self, time_series: np.ndarray, **kwargs):
        pass

    @abstractmethod
    def predict(self, horizon: int, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        pass


class BenchmarkExperiment:
    """Базовый класс для проведения экспериментов"""

    def __init__(self, experiment_name: str, results_dir: str = "results"):
        self.experiment_name = experiment_name
        self.results_dir = results_dir
        self.models = {}
        self.results = []
        self.metrics_history = []

        # Создаем директории для результатов
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "metrics"), exist_ok=True)

    def add_model(self, name: str, model: BaseForecaster):
        """Добавление модели в эксперимент"""
        self.models[name] = model

    def run_experiment(self, dataset_loader, series_ids: List[str] = None,
                       horizons: List[int] = None, cv_folds: int = 5):
        """Запуск эксперимента на наборе данных"""
        if horizons is None:
            horizons = [1, 3, 5, 10, 20]

        print(f"🚀 Запуск эксперимента: {self.experiment_name}")
        print(f"📊 Модели: {list(self.models.keys())}")
        print(f"🎯 Горизонты: {horizons}")
        print(f"📈 CV folds: {cv_folds}")

        # Загрузка данных
        dataset = dataset_loader.load_data(series_ids)

        for series_id, time_series in dataset.items():
            print(f"\n🔍 Обработка ряда: {series_id}")

            series_results = self._evaluate_series(
                series_id, time_series, horizons, cv_folds
            )

            self.results.extend(series_results)
            self._save_series_results(series_id, series_results)
            self._plot_series_results(series_id, time_series, series_results)

        self._save_final_report()

    def _evaluate_series(self, series_id: str, time_series: np.ndarray,
                         horizons: List[int], cv_folds: int) -> List[Dict]:
        """Оценка моделей на одном временном ряде"""
        series_results = []

        for horizon in horizons:
            print(f"  📐 Горизонт: {horizon}")

            for model_name, model in self.models.items():
                print(f"    🤖 Модель: {model_name}")

                try:
                    cv_metrics = self._cross_validate(
                        model, time_series, horizon, cv_folds
                    )

                    # Прогноз на всем наборе для визуализации
                    train_size = int(len(time_series) * 0.8)
                    train_data = time_series[:train_size]

                    # Клонируем модель для избежания side effects
                    model_clone = self._clone_model(model)
                    model_clone.fit(train_data)
                    forecasts = model_clone.predict(horizon)

                    result = {
                        'series_id': series_id,
                        'model_name': model_name,
                        'horizon': horizon,
                        'cv_metrics': cv_metrics,
                        'forecasts': forecasts.tolist(),
                        'train_size': train_size,
                        'timestamp': datetime.now().isoformat()
                    }

                    series_results.append(result)
                    self.metrics_history.append({
                        'series_id': series_id,
                        'model_name': model_name,
                        'horizon': horizon,
                        **cv_metrics
                    })

                    print(f"      ✅ RMSE: {cv_metrics['rmse']:.4f}")

                except Exception as e:
                    print(f"      ❌ Ошибка: {e}")
                    # Сохраняем информацию об ошибке
                    error_result = {
                        'series_id': series_id,
                        'model_name': model_name,
                        'horizon': horizon,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    series_results.append(error_result)

        return series_results

    def _cross_validate(self, model: BaseForecaster, time_series: np.ndarray,
                        horizon: int, folds: int) -> Dict[str, float]:
        """Кросс-валидация для одного ряда и горизонта"""
        from ..evaluation.cross_validation import TimeSeriesCrossValidator

        validator = TimeSeriesCrossValidator(folds=folds)
        metrics = validator.evaluate(model, time_series, horizon)
        return metrics

    def _clone_model(self, model: BaseForecaster) -> BaseForecaster:
        """Создание копии модели (упрощенная версия)"""
        # В реальной реализации нужно глубокое копирование
        # или пересоздание модели с теми же параметрами
        return model

    def _save_series_results(self, series_id: str, results: List[Dict]):
        """Сохранение результатов для одного ряда"""
        filename = f"{self.results_dir}/metrics/{self.experiment_name}_{series_id}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

    def _plot_series_results(self, series_id: str, time_series: np.ndarray,
                             results: List[Dict]):
        """Визуализация результатов для одного ряда"""
        from ..visualization.forecast_plotter import ForecastPlotter

        plotter = ForecastPlotter()
        plotter.plot_series_comparison(series_id, time_series, results,
                                       self.results_dir)

    def _save_final_report(self):
        """Сохранение финального отчета"""
        # Агрегация метрик
        metrics_df = pd.DataFrame(self.metrics_history)

        # Сохранение в разных форматах
        metrics_df.to_csv(f"{self.results_dir}/{self.experiment_name}_metrics.csv", index=False)
        metrics_df.to_excel(f"{self.results_dir}/{self.experiment_name}_metrics.xlsx", index=False)

        # Генерация отчета
        self._generate_summary_report(metrics_df)

    def _generate_summary_report(self, metrics_df: pd.DataFrame):
        """Генерация сводного отчета"""
        report = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'total_series': metrics_df['series_id'].nunique(),
            'total_models': metrics_df['model_name'].nunique(),
            'total_horizons': metrics_df['horizon'].nunique(),
            'summary_metrics': self._calculate_summary_metrics(metrics_df)
        }

        with open(f"{self.results_dir}/{self.experiment_name}_report.json", 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📊 Финальный отчет сохранен: {self.results_dir}/{self.experiment_name}_report.json")

    def _calculate_summary_metrics(self, metrics_df: pd.DataFrame) -> Dict:
        """Вычисление сводных метрик"""
        summary = {}

        # Метрики по моделям
        model_metrics = metrics_df.groupby('model_name').agg({
            'rmse': ['mean', 'std', 'min', 'max'],
            'mae': ['mean', 'std'],
            'mape': ['mean', 'std'],
            'smape': ['mean', 'std']
        }).round(4)

        summary['model_performance'] = model_metrics.to_dict()

        # Метрики по горизонтам
        horizon_metrics = metrics_df.groupby('horizon').agg({
            'rmse': 'mean',
            'mae': 'mean',
            'smape': 'mean'
        }).round(4)

        summary['horizon_performance'] = horizon_metrics.to_dict()

        return summary
