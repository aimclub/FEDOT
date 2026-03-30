# Промежуточный PR по TensorData

## Краткое описание

Главная цель — сделать `TensorData` реальным внутренним execution path, не ломая существующий API на `InputData`.

## Что вошло в PR

- стабилизация taxonomy, normalization и creator/backend rules для `TensorData`
- compatibility mapping между `InputData` и TensorData 
- явные  адаптеры `InputData -> TensorData` и `TensorData -> InputData`
-  entrypoint для адаптеров и новых типов в `ApiDataProcessor` и `DataPreprocessor`
- явные `Pipeline.fit_tensordata(...)` и `Pipeline.predict_tensordata(...)`
- явные `Fedot.predict_tensordata(...)`, `Fedot.predict_proba_tensordata(...)` и predefined-only `Fedot.fit_tensordata(...)`
- mirrored tests в `tests/core/data/...`, `tests/preprocessing/...`, `tests/core/pipelines/...`, `tests/api/...`

## Чего здесь нет

- полной composer-поддержки для `TensorData`
- generic TensorData composition
- TensorData fit path с `predefined_model='auto'`
- convergence для `fedot/industrial`

## Ключевый архитектурный смысл

Мы не переписываем FEDOT целиком под `TensorData`.
Вместо этого мы оставляем OOP-boundary "тонкими" и выносим 
в pure core normalization, compatibility mapping, bridge planning и runtime stage selection.

## Важное ограничение

`TensorData` fit пока поддержан только для predefined/runtime path-ов. 
Неподдерживаемые варианты отсекаются рано и с понятной ошибкой.

## Тестирование

В этой среде проверялись `python -m py_compile` для затронутых модулей и тестов. Полный `pytest` здесь не гонялся.

## Рекомендуемый порядок ревью

1. `fedot/core/repository/dataset_types.py`
2. `fedot/core/data/*`
3. `fedot/preprocessing/*`
4. `fedot/core/pipelines/*`
5. `fedot/api/api_utils/*`
6. `fedot/api/main.py`
7. mirrored tests в `tests/`
