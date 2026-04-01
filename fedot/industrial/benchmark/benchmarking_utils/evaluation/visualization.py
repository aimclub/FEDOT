# benchmark/evaluation/visualizations.py
import json
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class ResultVisualizer:
    """Визуализация результатов benchmarking"""

    def __init__(self, output_dir='results/plots'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Настройка стиля
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def plot_forecast_comparison(self, y_true: np.ndarray, forecasts: Dict[str, np.ndarray],
                                 model_names: List[str], series_name: str, horizon: int):
        """Сравнение прогнозов разных моделей"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # График 1: Полный ряд с прогнозами
        full_length = len(y_true) + horizon
        train_length = len(y_true) - horizon

        # Обучающая часть
        ax1.plot(range(train_length), y_true[:train_length],
                 'b-', linewidth=2, label='Train', alpha=0.8)

        # Тестовая часть
        ax1.plot(range(train_length, len(y_true)), y_true[train_length:],
                 'g-', linewidth=2, label='Test', alpha=0.8)

        # Прогнозы
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
        for i, (model_name, forecast) in enumerate(forecasts.items()):
            forecast_x = range(len(y_true), len(y_true) + len(forecast))
            ax1.plot(forecast_x, forecast,
                     color=colors[i], linewidth=2, marker='o', markersize=4,
                     label=f'{model_name} forecast')

        ax1.axvline(x=len(y_true), color='red', linestyle='--', alpha=0.7, label='Forecast start')
        ax1.set_title(f'Forecast Comparison: {series_name}\nHorizon: {horizon}',
                      fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # График 2: Увеличенный вид прогнозов
        zoom_start = max(0, len(y_true) - horizon * 2)

        ax2.plot(range(zoom_start, len(y_true)), y_true[zoom_start:],
                 'g-', linewidth=2, label='Actual', alpha=0.8)

        for i, (model_name, forecast) in enumerate(forecasts.items()):
            forecast_x = range(len(y_true), len(y_true) + len(forecast))
            ax2.plot(forecast_x, forecast,
                     color=colors[i], linewidth=2, marker='o', markersize=4,
                     label=f'{model_name}')

            # Ошибки прогноза
            actual_test = y_true[train_length:train_length + len(forecast)]
            errors = np.abs(forecast - actual_test)
            ax2.fill_between(forecast_x, forecast - errors, forecast + errors,
                             color=colors[i], alpha=0.2)

        ax2.set_title('Zoomed Forecast View with Error Ranges', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = self.output_dir / f'forecast_comparison_{series_name}_{horizon}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        return filename

    def plot_model_ranking(self, results_df: pd.DataFrame, metric: str = 'MASE'):
        """Ранжирование моделей по метрике"""
        plt.figure(figsize=(12, 8))

        # Группируем по моделям
        model_stats = results_df.groupby('model')[metric].agg(['mean', 'std']).sort_values('mean')

        # Столбчатая диаграмма с ошибками
        bars = plt.bar(range(len(model_stats)), model_stats['mean'],
                       yerr=model_stats['std'], capsize=5, alpha=0.7,
                       color=plt.cm.viridis(np.linspace(0, 1, len(model_stats))))

        plt.xlabel('Models')
        plt.ylabel(f'{metric} (lower is better)')
        plt.title(f'Model Ranking by {metric}', fontsize=14, fontweight='bold')
        plt.xticks(range(len(model_stats)), model_stats.index, rotation=45)
        plt.grid(True, alpha=0.3, axis='y')

        # Добавляем значения на столбцы
        for i, (bar, (model, stats)) in enumerate(zip(bars, model_stats.iterrows())):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{stats["mean"]:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        filename = self.output_dir / f'model_ranking_{metric}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        return filename

    def plot_metric_distribution(self, results_df: pd.DataFrame, metric: str = 'MASE'):
        """Распределение метрик по моделям"""
        plt.figure(figsize=(12, 8))

        # Box plot распределения метрик
        sns.boxplot(data=results_df, x='model', y=metric)
        plt.xticks(rotation=45)
        plt.title(f'Distribution of {metric} across Models', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = self.output_dir / f'metric_distribution_{metric}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        return filename

    def create_summary_report(self, results_df: pd.DataFrame, output_path: str):
        """Создание сводного отчета"""
        report = {
            'summary': {
                'total_series': len(results_df['series_id'].unique()),
                'total_models': len(results_df['model'].unique()),
                'evaluation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'model_performance': {},
            'best_models': {}
        }

        # Производительность по моделям
        metrics = ['MASE', 'sMAPE', 'RMSE', 'MAE']

        for metric in metrics:
            if metric in results_df.columns:
                model_ranking = results_df.groupby('model')[metric].mean().sort_values()
                report['model_performance'][metric] = model_ranking.to_dict()
                report['best_models'][metric] = model_ranking.index[0]

        # Сохраняем отчет
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        return report
