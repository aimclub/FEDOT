# Промежуточный PR по TensorData

## Краткое описание

Главная цель этого PR — сделать `TensorData` реальным внутренним execution path, не ломая существующий публичный API на `InputData`.

В этом срезе мы не переписываем FEDOT целиком под новый data model. Вместо этого мы последовательно добавляем:
- стабилизированный TensorData core
- явные compatibility bridges
- bridge-aware preprocessing
- tensor-aware runtime entrypoint'ы в `Pipeline` и `Fedot`
- tensor-aware objective boundary для split/evaluation path

## Что вошло в PR

- стабилизация taxonomy, normalization, creator/backend rules и file-source rules для `TensorData`
- compatibility mapping между legacy `InputData` data types и canonical TensorData view
- явные адаптеры `InputData -> TensorData` и `TensorData -> InputData`
- typed descriptor/preparation layer для legacy `InputData`
- entrypoint'ы для bridge boundary в `ApiDataProcessor` и `DataPreprocessor`
- явные `Pipeline.fit_tensordata(...)` и `Pipeline.predict_tensordata(...)`
- явные методы в `Fedot` для `fit_tensordata`, `predict_tensordata`, `predict_proba_tensordata`, `tune_tensordata`, `forecast_tensordata`, `get_metrics_tensordata`, `explain_tensordata`
- parity-логика для merge API/pipeline preprocessors после `fit_tensordata(...)`
- tensor-aware objective boundary через `DataSourceSplitter.build_tensordata(...)`
- нормализация extension data types для legacy runtime view и TensorData canonical view
- mirrored tests в `tests/core/data/...`, `tests/preprocessing/...`, `tests/core/pipelines/...`, 
- `tests/core/optimisers/objective/...`, `tests/extensions/...`, `tests/api/...`

## Чего здесь нет

- полной composer-поддержки для `TensorData`
- generic TensorData composition
- TensorData fit path с `predefined_model='auto'`
- глубокой tensor-native интеграции всех evaluation strategies
- convergence для `fedot/industrial`

## Ключевой архитектурный смысл

Мы оставляем OOP boundary-классы на своих местах:
- `Fedot` как facade
- `ApiDataProcessor` как adapter
- `DataPreprocessor` как OOP service
- `Pipeline` как runtime composite
- `DataSourceSplitter` как objective boundary

При этом decisions и compatibility concerns постепенно локализуются в отдельных слоях:
- normalization rules
- compatibility mapping
- bridge adapters
- service/runtime plans

Это делает TensorData path явным, предсказуемым и тестируемым, не превращая переход в большой rewrite.

## Поведенческие изменения

- TensorData path теперь поддержан как явный runtime flow, а не как скрытая внутренняя ветка.
- Неподдерживаемые TensorData fit scenarios по-прежнему отсекаются рано и с понятной ошибкой.
- Preprocessing и runtime path для TensorData проходят через контролируемые bridge boundaries.
- Objective split path теперь тоже может стартовать от TensorData через явный adapter, 
- а не через ручную конверсию в вызывающем коде.

## Тестирование

В этой среде для каждого среза проверялся `python -m py_compile` на затронутых модулях и mirrored tests.
Полный `pytest` здесь не гонялся, потому что в окружении отсутствует `pytest`, а часть runtime всё ещё зависит от отсутствующего `golem`.

## Риски и компромиссы

- Часть TensorDatа пока интегрированно параллельно с InputData.
- Это осознанный компромисс: сначала мы фиксируем безопасные границы совместимости, затем уже будем убирать 
- legacy-only paths глубже в runtime.
- Composer и industrial Ностаются за пределами этого PR, чтобы не раздувать скоуп.

## Рекомендуемый порядок ревью

1. `fedot/core/repository/dataset_types.py`
2. `fedot/core/data/*`
3. `fedot/preprocessing/*`
4. `fedot/core/pipelines/*`
5. `fedot/core/optimisers/objective/*`
6. `fedot/extensions/*`
7. `fedot/api/api_utils/*`
8. `fedot/api/main.py`
9. mirrored tests в `tests/`
