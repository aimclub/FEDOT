# benchmark/main.py
# !/usr/bin/env python3

import traceback

import numpy as np
import pandas as pd

from benchmark.benchmarking_utils.core.config import (TimeSeriesBenchmark, BenchmarkConfig)
from benchmark.benchmarking_utils.evaluation.metrics import ForecastEvaluator, ModelComparator
from benchmark.benchmarking_utils.evaluation.visualization import ResultVisualizer


def main():
    """Основная функция benchmarking"""

    # Конфигурация
    config = BenchmarkConfig()
    benchmark = TimeSeriesBenchmark(config)
    visualizer = ResultVisualizer()
    evaluator = ForecastEvaluator()
    comparator = ModelComparator()

    # Результаты будут собираться здесь
    all_results = []

    # Benchmarking по датасетам
    benchmarks_to_run = [
        ('m4', 'Yearly'),
        ('m4', 'Quarterly'),
        ('m4', 'Monthly'),
        ('monash', 'electricity'),
        ('monash', 'traffic')
    ]

    for benchmark_name, subset in benchmarks_to_run:
        print(f"\n{'=' * 60}")
        print(f"Running benchmark: {benchmark_name} - {subset}")
        print(f"{'=' * 60}")

        try:
            # Загрузка датасета
            dataset = benchmark.load_dataset(benchmark_name, subset)
            series_list = dataset['series']

            # Ограничение количества рядов для тестирования
            max_series = config.config['benchmarks'][benchmark_name].get('max_series_per_subset', 10)
            series_list = series_list[:max_series]

            # Прогон по каждому ряду
            for series_idx, series in enumerate(series_list):
                print(f"\nProcessing series {series_idx + 1}/{len(series_list)}")

                series_results = run_single_series_benchmark(
                    series, benchmark.models, evaluator, config
                )

                # Добавляем метаданные
                for result in series_results:
                    result.update({
                        'benchmark': benchmark_name,
                        'subset': subset,
                        'series_id': f"{benchmark_name}_{subset}_{series_idx}",
                        'series_length': len(series)
                    })

                all_results.extend(series_results)

                # Сохраняем промежуточные результаты
                if series_idx % 5 == 0:
                    save_intermediate_results(all_results, f"intermediate_{benchmark_name}_{subset}.csv")

        except Exception as e:
            print(f"Error processing {benchmark_name} - {subset}: {e}")
            traceback.print_exc()

    # Финальный анализ и визуализация
    if all_results:
        results_df = pd.DataFrame(all_results)
        create_final_report(results_df, visualizer, comparator)
    else:
        print("No results collected!")


def run_single_series_benchmark(series: np.ndarray, models: dict,
                                evaluator: ForecastEvaluator, config: BenchmarkConfig) -> list:
    """Benchmark для одного временного ряда"""
    results = []
    metrics = config.config['evaluation']['metrics']
    horizons = config.config['evaluation']['forecast_horizons']
    test_size = config.config['evaluation']['test_size']

    # Разделение на train/test
    split_point = int(len(series) * (1 - test_size))
    y_train = series[:split_point]
    y_test = series[split_point:]

    forecasts = {}

    for model_name, model_class in models.items():
        print(f"  Training {model_name}...")

        for horizon in horizons:
            try:
                # Создание и обучение модели
                model = model_class()
                model.fit(y_train)

                # Прогнозирование
                forecast = model.predict(horizon=horizon)

                # Обрезаем прогноз если нужно
                actual_horizon = min(horizon, len(y_test))
                forecast = forecast[:actual_horizon]
                y_true_actual = y_test[:actual_horizon]

                # Оценка
                metrics_values = evaluator.evaluate_forecast(
                    y_true_actual, forecast, y_train, metrics
                )

                # Сохраняем результаты
                result = {
                    'model': str(model),
                    'horizon': actual_horizon,
                    'forecast_length': len(forecast),
                    'train_length': len(y_train),
                    'test_length': len(y_true_actual)
                }
                result.update(metrics_values)
                results.append(result)

                # Сохраняем прогноз для визуализации
                if horizon == horizons[-1]:  # Сохраняем только для последнего горизонта
                    forecasts[str(model)] = forecast

            except Exception as e:
                print(f"    Error with {model_name} at horizon {horizon}: {e}")
                # Добавляем запись с NaN значениями
                result = {
                    'model': model_name,
                    'horizon': horizon,
                    'forecast': forecasts
                }
