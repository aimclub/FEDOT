# Example Prompts

## Route to `$fedot-pure-core-shell`

- "Вынеси decision rules из `Fedot`, не ломая публичный фасад."
- "Раздели preprocessing orchestrator и чистые merge rules."

## Route to `$fedot-safe-configs`

- "Убери `eval` из remote config parsing и сделай typed validation."
- "Перенеси defaulting и normalization параметров в один безопасный путь."

## Route to `$fedot-typed-domain-errors`

- "Заменим string flags и sentinel values на явные typed результаты."
- "Сделай domain error model для выбора backend и preset."

## Route to `$fedot-extension-contract`

- "Добавь внешний model manifest без правки нескольких внутренних списков."
- "Пересобери registry и smoke-test flow для extensions."

## Route to `$fedot-invariant-tests-review`

- "Проверь этот refactoring PR на риски и недостающие тесты."
- "Добавь tests для deterministic planner и normalization invariants."

## Composite Examples

- "Убери sentinel `'None'` из remote config и добавь coverage."  
  Primary: `$fedot-safe-configs`  
  Secondary: `$fedot-typed-domain-errors`, `$fedot-invariant-tests-review`

- "Вынеси typed planner из facade и проверь регрессии."  
  Primary: `$fedot-pure-core-shell`  
  Secondary: `$fedot-invariant-tests-review`