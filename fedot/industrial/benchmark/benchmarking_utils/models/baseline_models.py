# models/baseline_models.py
import numpy as np
import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA

from ..core.base_benchmark import BaseForecaster


class ARIMAForecaster(BaseForecaster):
    """Модель ARIMA как baseline"""

    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None

    def fit(self, time_series: np.ndarray, **kwargs):
        self.time_series = time_series
        try:
            self.model = ARIMA(time_series, order=self.order)
            self.fitted_model = self.model.fit()
        except Exception as e:
            # Fallback to simple differencing if ARIMA fails
            print(f"ARIMA fit failed: {e}, using simple forecast")
            self.fitted_model = None

    def predict(self, horizon: int, **kwargs) -> np.ndarray:
        if self.fitted_model is not None:
            forecast = self.fitted_model.forecast(steps=horizon)
            return forecast.values
        else:
            # Simple persistence forecast
            last_value = self.time_series[-1]
            return np.full(horizon, last_value)

    def get_model_info(self) -> dict:
        return {'name': 'ARIMA', 'order': self.order}


class ExponentialSmoothingForecaster(BaseForecaster):
    """Простое экспоненциальное сглаживание"""

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.last_value = None

    def fit(self, time_series: np.ndarray, **kwargs):
        self.last_value = time_series[-1]

    def predict(self, horizon: int, **kwargs) -> np.ndarray:
        forecasts = []
        current = self.last_value

        for _ in range(horizon):
            forecasts.append(current)
            # Simple persistence (можно улучшить)
            current = current

        return np.array(forecasts)

    def get_model_info(self) -> dict:
        return {'name': 'ExponentialSmoothing', 'alpha': self.alpha}


class ThetaForecaster(BaseForecaster):
    """Модель Theta для прогнозирования"""

    def fit(self, time_series: np.ndarray, **kwargs):
        self.time_series = time_series
        # Простая реализация Theta метода
        self.trend = self._compute_trend(time_series)

    def predict(self, horizon: int, **kwargs) -> np.ndarray:
        # Упрощенная реализация Theta
        last_trend = self.trend[-1] if len(self.trend) > 0 else 0
        forecasts = []

        for i in range(horizon):
            forecast = self.time_series[-1] + (i + 1) * last_trend
            forecasts.append(forecast)

        return np.array(forecasts)

    def _compute_trend(self, series: np.ndarray) -> np.ndarray:
        # Простое вычисление тренда через разности
        return np.diff(series)

    def get_model_info(self) -> dict:
        return {'name': 'Theta'}


class MLPForecaster(BaseForecaster):
    """Простая нейросетевая модель как baseline"""

    def __init__(self, hidden_size=50, num_layers=2, learning_rate=0.001):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.model = None
        self.window_size = 10

    def fit(self, time_series: np.ndarray, **kwargs):
        # Подготовка данных для обучения
        X, y = self._prepare_data(time_series)

        if len(X) == 0:
            self.model = None
            return

        # Создание модели
        self.model = self._build_model(X.shape[1])

        # Обучение
        self._train_model(X, y)

    def predict(self, horizon: int, **kwargs) -> np.ndarray:
        if self.model is None:
            return np.full(horizon, self._last_value)

        # Рекурсивное прогнозирование
        forecasts = []
        current_input = torch.tensor(self._last_window, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            for _ in range(horizon):
                prediction = self.model(current_input)
                forecasts.append(prediction.item())

                # Обновляем вход для следующего шага
                current_input = torch.cat([
                    current_input[:, 1:],
                    prediction.unsqueeze(0).unsqueeze(0)
                ], dim=1)

        return np.array(forecasts)

    def _prepare_data(self, series: np.ndarray):
        X, y = [], []

        for i in range(len(series) - self.window_size - 1):
            X.append(series[i:i + self.window_size])
            y.append(series[i + self.window_size])

        if len(X) > 0:
            self._last_window = series[-self.window_size:]
            self._last_value = series[-1]

        return np.array(X), np.array(y)

    def _build_model(self, input_size: int):
        layers = []
        layers.append(nn.Linear(input_size, self.hidden_size))
        layers.append(nn.ReLU())

        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(self.hidden_size, 1))

        return nn.Sequential(*layers)

    def _train_model(self, X, y, epochs=100):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = self.model(X_tensor)
            loss = criterion(predictions, y_tensor)
            loss.backward()
            optimizer.step()

    def get_model_info(self) -> dict:
        return {
            'name': 'MLP',
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers
        }
