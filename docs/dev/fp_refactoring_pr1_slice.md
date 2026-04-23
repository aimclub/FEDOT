# Первый PR: OOP Shell over Typed Pure Core

## В чем идея PR

В этом ПР сделана первая последовательная вертикальная часть плана рефакторинга:
оставлен общедоступный API ООП и объекты ядра, 
логику принятия решений вынесена в чистые функции,
валидацию и нормализацию параметров так же.

## Что поменялось

- `fedot/extensions`
  - extension contract
  - registry
  - operation discovery bridge
  - runtime adapter
  - typed extension parameter resolution
- `fedot/remote`
  - safe typed pipeline config parsing without `eval`
- `fedot/api`
  - typed run/service planning rules
  - extracted params/defaulting/recommendation/preset/assumption rules
  - `Fedot` facade still preserved as OOP shell
- `fedot/preprocessing`
  - source, merge and optional-preprocessing planning rules
- `fedot/core/repository`
  - typed operation query and pipeline operation split rules
- `fedot/core/pipelines`
  - pipeline preprocess/postprocess rules
  - pipeline node parameter normalization rules
- `fedot/core/operations`
  - operation parameter normalization/change-tracking rules
- `tests/`
  - mirrored tree for `api`, `core`, `extensions`, `preprocessing`, `remote`

## Архитектурный эффект

- Зоны влияния ООП остаются на месте.
- Скрытая логика ветвления и нормализации перенесена в небольшие чистые функции.
- Ожидаемые сбои на новых границах представлены более явно.
- Интеграция с внешней моделью больше не зависит от редактирования нескольких внутренних конфигураций.

## Что намерено не было сделано в этом PR

- рефактор индастриала
- рефактор CI
- работа над моделями и методами для фичей

## В каком порядке сомтреть
1. extension contract and runtime bridge
2. remote config safety changes
3. api/core/preprocessing pure-rule extractions
4. mirrored tests structure and new `pytest` markers
