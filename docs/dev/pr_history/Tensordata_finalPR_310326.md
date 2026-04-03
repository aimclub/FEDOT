# Итоговый PR по TensorData и industrial convergence

## Кратко

Этот PR завершает первый большой вертикальный срез TensorData-first refactoring и доводит `fedot/industrial`
до совместимого состояния с новой архитектурной линией.

Основная идея PR:
- `TensorData` становится явным внутренним execution path
- legacy `InputData` остаётся compatibility shell
- OOP facade/service/builder/composite классы сохраняются
- planning, normalization, dispatch и compatibility concerns выносятся в pure rules
- industrial слой начинает использовать те же typed plans и boundary contracts

## Что вошло в PR

### TensorData core и bridges
- стабилизация taxonomy и compatibility mapping для `TensorData`
- creator/backend normalization rules
- file-source и data-reader rules
- bridges `InputData -> TensorData` и `TensorData -> InputData`
- descriptor/preparation layer для legacy `InputData`

### API, preprocessing, pipeline и tuning
- explicit TensorData path в `ApiDataProcessor`
- bridge-aware preprocessing boundary
- explicit TensorData runtime entrypoint'ы в `Pipeline`
- explicit TensorData facade methods в `Fedot` для fit/predict/proba/tune/forecast/metrics/explain
- tensor-aware tuning и objective boundaries

### Extension/runtime compatibility
- manifest-driven extension contract
- extension discovery в repository/query path
- runtime adapter для extension-backed operations
- нормализация legacy и tensor data type views для extensions

### Industrial convergence
- typed rules для industrial data/task normalization
- typed rules для industrial init/context/loss/state planning
- typed dispatch/runtime plans для industrial strategy
- pure rules для `FedotIndustrial.main` вокруг predict/metrics/save/load/explain/history/fit/proba/finetune
- безопасное config defaulting и normalization для `ComputationalConfig` и `AutomlConfig`
- cleanup hidden mutation в industrial sampling и finetune paths

### Тестовая структура
- mirrored tests в `tests/` для новых pure modules и сохранённых OOP-shell entrypoints
- unit tests для planners и normalizers
- shell/facade tests для compatibility boundaries
- инварианты на idempotence, determinism, round-trip и boundary preservation там, где это важно

## Архитектурный смысл

Этот PR не переписывает FEDOT в чисто функциональном стиле. Он делает архитектуру двуслойной:
- внешний слой остаётся OOP-first
- внутренние правила становятся deterministic и тестируемыми

Практический эффект такой:
1. TensorData path становится явным, а не скрытым экспериментальным ответвлением.
2. Legacy compatibility остаётся под контролем через bridge boundary.
3. Industrial слой начинает сходиться с общей архитектурой проекта, а не жить по отдельным stringly-typed правилам.

## Что не входит в PR

- полная composer-поддержка для TensorData
- полная tensor-native интеграция всех evaluation strategies
- полный отказ от legacy `InputData`
- дальнейшая глубокая переработка `fedot/industrial` за пределами API/config/runtime boundary

## Поведенческие изменения

- TensorData поддержан как явный runtime flow в core и facade слоях
- preprocessing и objective path умеют работать через контролируемые TensorData bridges
- industrial API меньше зависит от неявных mode/status branching и shared mutable defaults
- config defaulting в industrial compute/automl слое стало детерминированным и безопасным для повторного использования

## Тестирование

В этой среде для срезов PR проверялся `python -m py_compile` на затронутых модулях и mirrored tests.
Полный `pytest` здесь не запускался, потому что в окружении отсутствует `pytest`, а часть runtime всё ещё зависит от отсутствующего `golem`.

## Риски и компромиссы

- В проекте всё ещё параллельно живут `InputData` и `TensorData`; это осознанный переходный компромисс.
- Не все runtime и evaluation paths нативно tensor-aware; часть из них пока работает через compatibility bridge.
- Industrial convergence закрывает API/config/runtime слой, но не претендует на полный rewrite всего industrial пакета.

## Рекомендуемый порядок ревью

1. `fedot/core/repository/*` и `fedot/core/data/*`
2. `fedot/preprocessing/*`
3. `fedot/core/pipelines/*` и `fedot/core/optimisers/objective/*`
4. `fedot/extensions/*`
5. `fedot/api/api_utils/*` и `fedot/api/*`
6. `fedot/industrial/api/utils/*`
7. `fedot/industrial/api/main.py` и `fedot/industrial/api/main_rules.py`
8. mirrored tests в `tests/`

## Follow-up

Следующий этап после этого PR:
- либо развивать composer и evaluation сторону TensorData
- либо продолжать industrial convergence глубже уровня API-shell
- но уже без возврата к скрытым conversions и shared mutable config flow
