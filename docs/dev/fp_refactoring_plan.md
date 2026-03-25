# План OOP-first refactoring с подготовкой к FP-informed архитектуре

## Summary

Первая волна рефакторинга сохраняет ключевые OOP-абстракции в `fedot/api` и `fedot/core` как публичный и координирующий слой, но выносит вычислительную, валидационную и selection-логику в pure core. Идея не в том, чтобы “сломать” существующий `Facade/Builder/Composite/Strategy` дизайн, а в том, чтобы сделать его тоньше, типобезопаснее и лучше совместимым с дальнейшей FP-интеграцией.

## OOP boundaries to preserve

- В `fedot/api` сохраняются `Fedot`, `FedotBuilder`, `ApiDataProcessor`, `ApiComposer`, `PredefinedModel`, `ApiParamsRepository`, `ApiParams`, `InputAnalyser`, assumptions/preset/filter builders и handlers как OOP-координаторы и boundary-объекты.
- В `fedot/core` сохраняются `PipelineNode`, `Pipeline`, `PipelineBuilder`, `PipelineTemplate`, `PipelineAdapter`, factory-слой, operation hierarchy, `EvaluationStrategy`, `Composer`, `ComposerBuilder`, objective/splitter abstractions.
- Правило рефакторинга: классы владеют lifecycle и orchestration, а правила выбора, валидация, трансформации и фильтрация выносятся в typed pure modules.

## First-wave implementation focus

1. Стабилизировать OOP API-слой через typed requests/results/specs без ломки `Facade/Builder`.
2. Вынести assumptions/preset/filter rules в отдельный pure core при сохранении текущих strategy/builder классов.
3. Выделить preprocessing plan/state и сократить неявный mutable state внутри preprocessor-а.
4. Разделить repository IO и pure parsing/filtering/query logic.
5. Ввести единый extension contract для внешних моделей без правки нескольких внутренних конфигов.
6. Переписать remote config parsing на безопасную typed модель без `eval` и sentinel `'None'`.

## External model contract

- Канонические сущности: `ExtensionManifest`, `ExternalModelSpec`, `ModelCapabilities`, `ModelFactory`, `ModelHyperparamsSchema`, `ExtensionError`.
- Канонический путь интеграции:
  `create manifest -> validate/register -> smoke test`.
- Новый contract должен быть OOP-friendly для пользователей и LLM-agent-friendly для автоматизации.
- Legacy JSON-репозитории остаются поддерживаемым boundary-слоем, но не рекомендуемым основным механизмом расширения.

## Test strategy

- Новая каноническая тестовая структура: `tests/`, зеркалящая `fedot/`.
- Тип теста выражается через pytest markers, а не через имя директории.
- Для OOP-координаторов обязательны service/facade tests.
- Для pure collaborators обязательны unit/property tests.
- Первые mirrored-кластеры: `tests/extensions`, затем `tests/api`, `tests/core`, `tests/preprocessing`, `tests/remote`.
