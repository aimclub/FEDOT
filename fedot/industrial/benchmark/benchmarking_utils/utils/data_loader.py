# utils/data_loader.py
import os
from typing import Dict, List

import numpy as np


class BenchmarkDataLoader:
    """Загрузчик данных бенчмарков M4 и Monash"""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self._ensure_directories()

    def _ensure_directories(self):
        """Создание необходимых директорий"""
        os.makedirs(os.path.join(self.data_dir, "m4"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "monash"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "results"), exist_ok=True)

    def load_m4_data(self, frequency: str = "Hourly") -> Dict[str, np.ndarray]:
        """
        Загрузка данных M4 benchmark
        frequency: 'Hourly', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly'
        """
        # Здесь будет реализация загрузки M4
        # Для начала создадим synthetic данные для тестирования
        return self._create_synthetic_m4_data(frequency)

    def load_monash_data(self, dataset_name: str = "tourism_monthly") -> Dict[str, np.ndarray]:
        """
        Загрузка данных Monash forecasting repository
        """
        return self._create_synthetic_monash_data(dataset_name)

    def _create_synthetic_m4_data(self, frequency: str) -> Dict[str, np.ndarray]:
        """Создание synthetic M4-like данных для тестирования"""
        np.random.seed(42)

        # Разные категории временных рядов
        categories = {
            'trend_seasonal': self._generate_trend_seasonal_series(100),
            'seasonal': self._generate_seasonal_series(100),
            'trend': self._generate_trend_series(100),
            'noisy': self._generate_noisy_series(100),
            'complex': self._generate_complex_series(100)
        }

        return categories

    def _create_synthetic_monash_data(self, dataset_name: str) -> Dict[str, np.ndarray]:
        """Создание synthetic Monash-like данных"""
        np.random.seed(42)

        datasets = {
            'tourism_monthly': self._generate_tourism_like_series(50),
            'electricity_hourly': self._generate_electricity_like_series(50),
            'traffic_weekly': self._generate_traffic_like_series(50),
            'covid_daily': self._generate_covid_like_series(50)
        }

        return datasets.get(dataset_name, {})

    def _generate_trend_seasonal_series(self, n_series: int) -> List[np.ndarray]:
        """Генерация рядов с трендом и сезонностью"""
        series_list = []
        for i in range(n_series):
            t = np.linspace(0, 10, 200)
            trend = 0.1 * t
            seasonal = 2 * np.sin(2 * np.pi * t) + 1 * np.sin(4 * np.pi * t)
            noise = 0.5 * np.random.normal(size=len(t))
            series = trend + seasonal + noise
            series_list.append(series)
        return series_list

    def _generate_seasonal_series(self, n_series: int) -> List[np.ndarray]:
        """Генерация чисто сезонных рядов"""
        series_list = []
        for i in range(n_series):
            t = np.linspace(0, 8, 150)
            seasonal = 3 * np.sin(2 * np.pi * t) + 1.5 * np.cos(4 * np.pi * t)
            noise = 0.3 * np.random.normal(size=len(t))
            series = seasonal + noise + 10  # Добавляем константу
            series_list.append(series)
        return series_list

    def _generate_trend_series(self, n_series: int) -> List[np.ndarray]:
        """Генерация рядов с трендом"""
        series_list = []
        for i in range(n_series):
            t = np.linspace(0, 5, 100)
            trend = 0.5 * t + 2 * np.sin(0.5 * t)  # Нелинейный тренд
            noise = 0.4 * np.random.normal(size=len(t))
            series = trend + noise
            series_list.append(series)
        return series_list

    def _generate_noisy_series(self, n_series: int) -> List[np.ndarray]:
        """Генерация зашумленных рядов"""
        series_list = []
        for i in range(n_series):
            t = np.linspace(0, 3, 80)
            signal = 2 * np.sin(3 * t)  # Слабый сигнал
            noise = 2 * np.random.normal(size=len(t))  # Сильный шум
            series = signal + noise + 5
            series_list.append(series)
        return series_list

    def _generate_complex_series(self, n_series: int) -> List[np.ndarray]:
        """Генерация сложных рядов с изменяющимся поведением"""
        series_list = []
        for i in range(n_series):
            t = np.linspace(0, 12, 300)
            # Комбинация разных паттернов
            part1 = 2 * np.sin(2 * np.pi * t[:100])  # Сезонность
            part2 = 0.1 * t[100:200] + 1 * np.sin(3 * np.pi * t[100:200])  # Тренд + сезонность
            part3 = 3 * np.exp(-0.1 * (t[200:] - 8)) * np.sin(4 * np.pi * t[200:])  # Затухание
            series = np.concatenate([part1, part2, part3])
            series += 0.5 * np.random.normal(size=len(series))
            series_list.append(series)
        return series_list

    # Monash-specific генераторы
    def _generate_tourism_like_series(self, n_series: int) -> List[np.ndarray]:
        """Туристические данные (месячные)"""
        series_list = []
        for i in range(n_series):
            t = np.linspace(0, 5, 60)  # 5 лет месячных данных
            seasonal = 10 * np.sin(2 * np.pi * t) + 5 * np.cos(4 * np.pi * t)
            trend = 0.2 * t
            noise = 2 * np.random.normal(size=len(t))
            series = seasonal + trend + noise + 20
            series_list.append(series)
        return series_list

    def _generate_electricity_like_series(self, n_series: int) -> List[np.ndarray]:
        """Электричество (часовые данные)"""
        series_list = []
        for i in range(n_series):
            t = np.linspace(0, 30, 720)  # 30 дней часовых данных
            # Суточная и недельная сезонность
            daily = 50 * np.sin(2 * np.pi * t / 24)
            weekly = 20 * np.sin(2 * np.pi * t / (24 * 7))
            noise = 10 * np.random.normal(size=len(t))
            series = daily + weekly + noise + 100
            series_list.append(series)
        return series_list
