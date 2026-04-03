# Как использовать FEDOT skillset

Этот документ описывает, как команде пользоваться набором FEDOT-скиллов при реальных задачах в репозитории.

## Где лежат исходники и установка

Исходники скиллов находятся в:

- `.codex/skills-src/`

Установленная копия для Codex находится в:

- `C:\Users\user\.codex\skills`

Синхронизация делается командой:

```powershell
powershell -ExecutionPolicy Bypass -File .\.codex\scripts\sync-skills.ps1 -SourceRoot .\.codex\skills-src -TargetRoot C:\Users\user\.codex\skills
```

## Состав skillset

### Базовые рабочие скиллы

- `$fedot-pure-core-shell` — вынос pure core из OOP shell при сохранении публичных границ
- `$fedot-invariant-tests-review` — выбор тестового слоя, инварианты, review refactoring PR
- `$fedot-safe-configs` — безопасный parse/validate/normalize/default flow
- `$fedot-extension-contract` — manifest, registry, smoke-test и runtime path для extensions
- `$fedot-typed-domain-errors` — явные domain state, requests/results и structured errors
- `$fedot-refactor-router` — роутинг задач между специализированными скиллами

### Командный meta-skill

- `$fedot-skillset-guide` — выбор lead skill, companion skills и порядок работы

## Быстрый порядок работы

### 1. Если задача неочевидная

Начать с:

- `$fedot-skillset-guide`
- при необходимости `$fedot-refactor-router`

Нужно получить короткий ответ:

- какой lead skill
- какие companion skills
- в каком порядке работать

### 2. Если задача уже понятна

Сразу выбирать один ведущий скилл и максимум два companion.

Нормальная композиция для большинства задач:

- один lead skill
- ноль, один или два companion skills

Не нужно подключать все скиллы одновременно.

### 3. После изменения архитектуры

Почти всегда заканчивать через:

- `$fedot-invariant-tests-review`

Это помогает не потерять:

- shell-level regression coverage
- direct helper tests
- invariants для deterministic logic

## Типовые комбинации

### Refactor facade или coordinator

- Lead: `$fedot-pure-core-shell`
- Companion: `$fedot-invariant-tests-review`

### Безопасный config flow

- Lead: `$fedot-safe-configs`
- Companion: `$fedot-typed-domain-errors`
- Companion: `$fedot-invariant-tests-review`

### Extension integration

- Lead: `$fedot-extension-contract`
- Companion: `$fedot-invariant-tests-review`

### Явные state и errors

- Lead: `$fedot-typed-domain-errors`
- Companion: `$fedot-pure-core-shell`
- Companion: `$fedot-invariant-tests-review`

### Неясная mixed-задача

- Lead: `$fedot-refactor-router`
- Затем переход к одному из специализированных скиллов

## Когда какой скилл выбирать по пакетам

- `fedot/api` — чаще всего `$fedot-pure-core-shell`, `$fedot-typed-domain-errors`, `$fedot-invariant-tests-review`
- `fedot/core` — чаще всего `$fedot-pure-core-shell`, `$fedot-typed-domain-errors`, `$fedot-invariant-tests-review`
- `fedot/preprocessing` — чаще всего `$fedot-pure-core-shell`, `$fedot-typed-domain-errors`, `$fedot-invariant-tests-review`
- `fedot/remote` — чаще всего `$fedot-safe-configs`, `$fedot-typed-domain-errors`, `$fedot-invariant-tests-review`
- `fedot/extensions` — чаще всего `$fedot-extension-contract`, `$fedot-invariant-tests-review`, `$fedot-typed-domain-errors`

## Минимальный рабочий workflow для задачи

1. Коротко классифицировать задачу одной фразой.
2. Выбрать lead skill.
3. Добавить только нужные companion skills.
4. Выполнить изменение вертикальным срезом, не размывая PR.
5. Добавить mirrored tests в тот же проход.
6. При изменении skillset прогнать smoke-набор.

## Smoke и сопровождение skillset

Smoke-prompts лежат в:

- `.codex/skills-src/smoke-prompts.md`

Командный smoke-checklist лежит в:

- `.codex/skills-smoke-suite.md`

Их стоит использовать, когда:

- изменили `SKILL.md`
- переписали `description`
- поменяли routing между скиллами
- обновили `agents/openai.yaml`

## Практическое правило

Если задача затрагивает и архитектуру, и типизацию, и тесты, не пытаться решить все одним «универсальным» навыком. Сначала выбрать главный угол атаки, а остальные concerns подключать как companion.