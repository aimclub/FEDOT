from typing import Dict, List, Optional

import numpy as np


class MonashDatasetLoader:
    """Загрузчик данных Monash Forecasting Repository"""

    def __init__(self, data_path: str = "data/monash"):
        self.data_path = data_path

    def load_data(self, dataset_names: Optional[List[str]] = None,
                  max_series_per_dataset: int = 50) -> Dict[str, np.ndarray]:
        """
        Загрузка данных из Monash Repository
        """
        if dataset_names is None:
            dataset_names = ['tourism_monthly', 'm4_monthly', 'cif_2016']

        all_series = {}

        for dataset in dataset_names:
            dataset_series = self._load_dataset(dataset, max_series_per_dataset)
            all_series.update(dataset_series)

        print(f"📊 Загружено {len(all_series)} рядов из Monash Repository")
        return all_series

    def _load_dataset(self, dataset_name: str, max_series: int) -> Dict[str, np.ndarray]:
        """Загрузка конкретного набора данных"""
        # Заглушка - в реальности загрузка из файлов
        series_dict = {}

        for i in range(min(20, max_series)):
            series_id = f"Monash_{dataset_name}_{i + 1:03d}"
            length = np.random.randint(50, 300)

            if 'monthly' in dataset_name:
                series = self._generate_monthly_series(length)
            elif 'quarterly' in dataset_name:
                series = self._generate_quarterly_series(length)
            else:
                series = self._generate_generic_series(length)

            series_dict[series_id] = series

        return series_dict

    def _generate_monthly_series(self, length: int) -> np.ndarray:
        t = np.arange(length)
        trend = 0.01 * t
        seasonal = 2.5 * np.sin(2 * np.pi * t / 12) + 1.2 * np.cos(2 * np.pi * t / 6)
        noise = 0.1 * np.random.normal(size=length)
        return 40 + trend + seasonal + noise

    def _generate_quarterly_series(self, length: int) -> np.ndarray:
        t = np.arange(length)
        trend = 0.02 * t
        seasonal = 1.8 * np.sin(2 * np.pi * t / 4)
        noise = 0.08 * np.random.normal(size=length)
        return 25 + trend + seasonal + noise

    def _generate_generic_series(self, length: int) -> np.ndarray:
        t = np.arange(length)
        trend = 0.005 * t
        seasonal = 1.5 * np.sin(2 * np.pi * t / 7)
        noise = 0.12 * np.random.normal(size=length)
        return 35 + trend + seasonal + noise
