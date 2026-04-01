# benchmark/core.py
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

warnings.filterwarnings('ignore')


class BenchmarkConfig:
    """Конфигурация benchmarking системы"""

    def __init__(self, config_path: Optional[str] = None):
        self.default_config = {
            'benchmarks': {
                'm4': {
                    'path': './data/m4',
                    'subsets': ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly'],
                    'max_series_per_subset': 100  # Для тестирования
                },
                'monash': {
                    'path': './data/monash',
                    'subsets': ['electricity', 'traffic', 'solar', 'exchange_rate'],
                    'max_series_per_subset': 50
                }
            },
            'models': {
                'our_models': ['OKHSForecaster', 'RKBSClassifier'],
                'baseline_models': ['ARIMA', 'ETS', 'Theta', 'Naive'],
                'advanced_models': ['DeepAR', 'NBEATS', 'Transformer', 'TFT']
            },
            'evaluation': {
                'metrics': ['MASE', 'sMAPE', 'RMSE', 'MAE', 'MAPE'],
                'test_size': 0.2,
                'cv_folds': 3,
                'forecast_horizons': [1, 3, 6, 12, 24]
            },
            'training': {
                'max_training_time': 3600,  # seconds
                'device': 'auto'
            }
        }

        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                self._update_config(self.default_config, user_config)

        self.config = self.default_config

    def _update_config(self, default, user):
        """Рекурсивное обновление конфигурации"""
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._update_config(default[key], value)
            else:
                default[key] = value


class TimeSeriesBenchmark:
    """Основной класс для benchmarking временных рядов"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}
        self.models = {}
        self.datasets = {}

        # Инициализация моделей
        self._initialize_models()

        # Создание директорий для результатов
        self._create_directories()

    def _initialize_models(self):
        """Инициализация всех моделей для сравнения"""
        from .models.our_models import OKHSForecasterTorch, RKBSClassifierTorch
        from .models.baseline_models import ARIMAModel, ETSModel, ThetaModel, NaiveModel
        from .models.advanced_models import DeepARModel, NBEATSModel, TransformerModel, TFTModel

        # Наши модели
        self.models['OKHSForecaster'] = OKHSForecasterTorch
        self.models['RKBSClassifier'] = RKBSClassifierTorch

        # Базовые модели
        self.models['ARIMA'] = ARIMAModel
        self.models['ETS'] = ETSModel
        self.models['Theta'] = ThetaModel
        self.models['Naive'] = NaiveModel

        # Продвинутые модели
        self.models['DeepAR'] = DeepARModel
        self.models['NBEATS'] = NBEATSModel
        self.models['Transformer'] = TransformerModel
        self.models['TFT'] = TFTModel

    def _create_directories(self):
        """Создание структуры директорий"""
        directories = [
            'results/tables',
            'results/plots/forecasts',
            'results/plots/comparisons',
            'results/logs',
            'data/m4',
            'data/monash'
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def load_dataset(self, benchmark_name: str, subset: str):
        """Загрузка датасета"""
        dataset_path = Path(self.config.config['benchmarks'][benchmark_name]['path']) / subset

        if benchmark_name == 'm4':
            return self._load_m4_dataset(dataset_path, subset)
        elif benchmark_name == 'monash':
            return self._load_monash_dataset(dataset_path, subset)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

    def _load_m4_dataset(self, dataset_path: Path, subset: str):
        """Загрузка M4 датасета"""
        # Здесь будет реализация загрузки M4
        # Временно создаем synthetic данные для тестирования
        return self._create_synthetic_data(100, subset)

    def _load_monash_dataset(self, dataset_path: Path, subset: str):
        """Загрузка Monash датасета"""
        # Здесь будет реализация загрузки Monash
        # Временно создаем synthetic данные для тестирования
        return self._create_synthetic_data(50, subset)

    def _create_synthetic_data(self, n_series: int, pattern: str):
        """Создание synthetic данных для тестирования"""
        series_list = []

        for i in range(n_series):
            if pattern == 'Yearly':
                length = 20 + np.random.randint(10)
            elif pattern == 'Quarterly':
                length = 80 + np.random.randint(40)
            elif pattern == 'Monthly':
                length = 240 + np.random.randint(120)
            elif pattern == 'Weekly':
                length = 520 + np.random.randint(260)
            elif pattern == 'Daily':
                length = 1825 + np.random.randint(365)
            else:  # Hourly
                length = 8760 + np.random.randint(8760)

            # Создаем ряд с определенным паттерном
            if pattern in ['Yearly', 'Quarterly', 'Monthly']:
                # Сезонные ряды
                t = np.linspace(0, 4 * np.pi, length)
                trend = 0.01 * np.arange(length)
                seasonality = np.sin(t) + 0.5 * np.sin(2 * t)
                noise = 0.1 * np.random.normal(size=length)
                series = trend + seasonality + noise
            else:
                # Более сложные паттерны для高频 данных
                t = np.linspace(0, 8 * np.pi, length)
                series = (np.sin(t) + 0.3 * np.sin(3 * t) +
                          0.1 * np.sin(7 * t) + 0.05 * np.random.normal(size=length))

            series_list.append(series)

        return {
            'name': f'synthetic_{pattern}',
            'series': series_list,
            'frequency': pattern,
            'metadata': {'synthetic': True}
        }
