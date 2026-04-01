# datasets/m4_loader.py
import os
from typing import Dict, List, Optional

import numpy as np


class M4DatasetLoader:
    """Загрузчик данных M4 Competition"""

    def __init__(self, data_path: str = "data/m4"):
        self.data_path = data_path
        self.frequency_map = {
            'Yearly': 1, 'Quarterly': 4, 'Monthly': 12,
            'Weekly': 52, 'Daily': 365, 'Hourly': 24
        }

    def load_data(self, series_ids: Optional[List[str]] = None,
                  frequency: str = 'Monthly', max_series: int = 100) -> Dict[str, np.ndarray]:
        """
        Загрузка данных M4

        Args:
            series_ids: список ID рядов для загрузки (None = все)
            frequency: частота данных
            max_series: максимальное число рядов для загрузки
        """
        # В реальной реализации здесь будет загрузка из файлов M4
        # Для демонстрации генерируем синтетические данные

        if not os.path.exists(self.data_path):
            print("⚠️  Реальные данные M4 не найдены, используем синтетические данные")
            return self._generate_synthetic_data(max_series, frequency)

        return self._load_real_m4_data(series_ids, frequency, max_series)

    def _generate_synthetic_data(self, n_series: int, frequency: str) -> Dict[str, np.ndarray]:
        """Генерация синтетических данных для тестирования"""
        series_dict = {}

        for i in range(n_series):
            series_id = f"M4_{frequency}_{i + 1:03d}"

            if frequency == 'Yearly':
                length = np.random.randint(20, 50)
                series = self._generate_yearly_series(length)
            elif frequency == 'Quarterly':
                length = np.random.randint(40, 100)
                series = self._generate_quarterly_series(length)
            elif frequency == 'Monthly':
                length = np.random.randint(60, 200)
                series = self._generate_monthly_series(length)
            else:
                length = np.random.randint(100, 500)
                series = self._generate_generic_series(length)

            series_dict[series_id] = series

        print(f"📊 Сгенерировано {len(series_dict)} синтетических рядов ({frequency})")
        return series_dict

    def _generate_yearly_series(self, length: int) -> np.ndarray:
        """Генерация годовых данных"""
        t = np.arange(length)
        trend = 0.05 * t
        noise = 0.1 * np.random.normal(size=length)
        return 10 + trend + noise

    def _generate_quarterly_series(self, length: int) -> np.ndarray:
        """Генерация квартальных данных"""
        t = np.arange(length)
        trend = 0.02 * t
        seasonal = 2 * np.sin(2 * np.pi * t / 4)
        noise = 0.1 * np.random.normal(size=length)
        return 20 + trend + seasonal + noise

    def _generate_monthly_series(self, length: int) -> np.ndarray:
        """Генерация месячных данных"""
        t = np.arange(length)
        trend = 0.01 * t
        seasonal = 3 * np.sin(2 * np.pi * t / 12)
        noise = 0.1 * np.random.normal(size=length)
        return 30 + trend + seasonal + noise

    def _generate_generic_series(self, length: int) -> np.ndarray:
        """Генерация общих временных рядов"""
        t = np.arange(length)
        trend = 0.005 * t
        seasonal = 2 * np.sin(2 * np.pi * t / 7) + 1.5 * np.sin(2 * np.pi * t / 30)
        noise = 0.15 * np.random.normal(size=length)
        return 25 + trend + seasonal + noise

    def _load_real_m4_data(self, series_ids: Optional[List[str]],
                           frequency: str, max_series: int) -> Dict[str, np.ndarray]:
        """Загрузка реальных данных M4 (заглушка)"""
        # Здесь должна быть реализация загрузки из CSV файлов M4
        print("📥 Загрузка реальных данных M4...")
        return self._generate_synthetic_data(min(50, max_series), frequency)
