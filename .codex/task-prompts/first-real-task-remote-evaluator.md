# Первый реальный task prompt для FEDOT skillset

## Цель

Живое изменение в кодовой базе, которое использует несколько FEDOT-скиллов по назначению.

## Рекомендуемый набор скиллов

- Lead: `$fedot-safe-configs`
- Companion: `$fedot-typed-domain-errors`
- Companion: `$fedot-invariant-tests-review`
- Если маршрутизация вызывает сомнение, сначала кратко использовать `$fedot-refactor-router`

## Почему выбран именно этот участок

Кандидат: `fedot/remote/remote_evaluator.py`

Почему это хороший первый живой кейс:

- в `RemoteTaskParams` используется stringly-typed `mode`
- несколько полей выражены через размытые `Optional[...]`
- `_get_config(...)` смешивает сериализацию значений и config-building
- в `tests/remote/` нет отдельного mirrored rule-test для этого модуля
- задача естественно требует и safe-config thinking, и explicit domain modeling, и новых тестов

## Готовый prompt

```text
Use $fedot-safe-configs, $fedot-typed-domain-errors, and $fedot-invariant-tests-review for a real FEDOT refactor in `fedot/remote/remote_evaluator.py`.

Task:
Refactor remote evaluator config preparation and task params handling so that the flow becomes safer and more explicit without breaking the current public behavior.

Focus areas:
1. Reduce stringly-typed and implicit state in `RemoteTaskParams`, especially around `mode` and config-related fields.
2. Separate pure config-building or normalization rules from the effectful evaluator shell where it improves clarity.
3. Keep public entry points such as `RemoteEvaluator`, `init_data_for_remote_execution`, and the external usage pattern compatible unless a very small compatibility wrapper is needed.
4. Add mirrored tests under `tests/remote` for the extracted or clarified behavior.

Requirements:
- Preserve current external behavior unless a failing edge case clearly justifies a safe compatibility-preserving adjustment.
- Avoid introducing `eval`, sentinel values, or generic exceptions for expected bad input.
- Prefer explicit typed records, enums, or structured domain results where they clarify the workflow.
- Keep I/O, client interaction, and singleton lifecycle at the shell boundary.
- Add tests for happy path plus at least one invalid or ambiguous input path.

Helpful local context:
- `fedot/remote/remote_evaluator.py`
- `tests/remote/test_pipeline_run_config.py`
- `docs/dev/fp_refactoring_plan.md`
- `docs/dev/fp_refactoring_pr1_slice.md`
- `docs/dev/Набор паттернов для создания FP-informed architecture.md`

Expected outcome:
- cleaner remote evaluator parameter/config flow
- more explicit domain semantics for remote task setup
- new or updated tests in `tests/remote`
- brief explanation of which parts became shell and which became pure rules
```

## Что должен проверить исполнитель

1. Не сломан ли внешний путь использования `RemoteEvaluator`.
2. Не размазан ли новый parse/normalize flow по нескольким функциям.
3. Появился ли явный typed результат или enum там, где раньше были строки или неявные состояния.
4. Добавлены ли tests именно под `tests/remote`, а не только косвенная проверка через другие модули.

## Возможный follow-up после этой задачи

Если refactor окажется удачным, следующим похожим кандидатом взять `fedot/api/api_utils/params.py` или связанный planner/rules слой в `fedot/api/api_utils`.