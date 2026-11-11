# FEDOT / Industrial: синхронизация полного перехода на TensorData

Дата: 5 ноября 2026 года. Аналитическая записка; код и патчи не изменялись.

Уточнение после реструктуризации операций в тот же день: канонические классы перенесены
в `core/operations/strategies/` и `core/operations/implementations/`.
Уточнение от 6 ноября: старые модули удалены полностью; Industrial обновит
свои импорты при синхронизации веток. Переэкспорты в FEDOT не сохраняются.
Результаты применимости текстовых patches в разделе 5 относятся к состоянию **до** переезда.
При синхронизации нужно перенести изменения по смыслу в новые канонические файлы.
Functional fixes merger/cache/CV не применялись.
Фактическая карта: [реструктуризация операций](fedot_operations_restructuring_2026_11_05.md).

## 1. Цель нашей работы и главный вывод

Цель — завершить переход **всего FEDOT** на TensorData: модели, преобразования,
полноценный поиск пайплайнов, оценивание, настройка и финальное обучение, включая
поддерживаемые GPU-сценарии. Реструктуризация файлов — промежуточное упорядочивание,
не основная цель. Tabular classification/regression могут быть первым acceptance
gate, но не заменяют конечный охват временных рядов и остальных поддерживаемых задач.

В проверенном Industrial **не обнаружена альтернативная реализация TensorData
или готовая замена полного поиска FEDOT**, из-за которой пришлось бы выбросить
наши модельные адаптации и переписать ядро. Основные контракты Ильи уже входят
в нашу базу. Industrial строит интеграцию и собственные поисковые механизмы поверх них.

Оценка: продолжать наш переход можно. Наибольший риск повторной работы — не в
разнице `features`/`predict` или именах файлов, а в независимом проектировании:

1. возможностей операций и допустимости графов;
2. преобразований, меняющих число объектов, target и индексы;
3. владения оцениванием, кешем, повторными попытками и исполнителями.

Эти границы нужно согласовывать до реализации соответствующих подсистем.
Остальные обнаруженные расхождения преимущественно локальны для адаптеров.

## 2. Проверенные версии и пределы вывода

| Компонент | Проверенный снимок |
| --- | --- |
| Наша ветка | `fedot_1.0.0_new`, HEAD `04094a67ab9bcb21fd6f274ee9f6a40450be39d1` + текущие незакоммиченные изменения TensorData-only моделей |
| Общая база FEDOT | `d1875e7a1ba49c94d13c51d97ec78829bf459ca8`, PR #1462 |
| Industrial main | `8fb8d2303442f554e30b2070df71c2b19e3108f3`, PR #318, принят 5 октября |
| Industrial PR #319 | Ветка `industrial_release_1.0`, `d631c7b56aac0602c09444eb57175fa52ead26db`, открытый PR; снимок от 28 октября |

PR и SHA сверены по GitHub API; код Industrial прочитан из отдельного временного
bare-репозитория `/tmp/fedot-overlap-review-53vvde/industrial.git`. Это временный
кеш исследования, не источник для будущей синхронизации: ниже даны постоянные ссылки.
Локальный `/home/artem/PycharmProjects/Fedot.Industrial` не переключался и не обновлялся.

Сделан адресный анализ контрактов, поисковых адаптеров и подготовленных FEDOT-патчей.
Полного интеграционного, GPU- или распределённого прогона совместной версии не было.
Оценки объёма переделки ниже — инженерные выводы по этим снимкам, не результаты
полной проверки совместимости и не гарантия будущего состояния PR.

Связанные материалы:

- [План TensorData search от 30 сентября](tensordata_full_pipeline_search_plan.md).
- [Карта моделей и операций](fedot_operations_data_formats_2026_10_09.md).
- [Предложение по структуре](fedot_operations_structure_proposal_2026_10_09.md).

Старый план search содержит исторические ссылки на `ModelEnginePlan`. Этот механизм
уже удалён после обсуждения. Не восстанавливать его механически и не вводить
rules/plan для каждой модели: отдельное планирование нужно для содержательного
этапа выбора пула/устройства/исполнения, если оно действительно требуется.

## 3. Что уже общее, а что остаётся нашей незавершённой работой

### Общий фундамент уже принят

[FEDOT PR #1462](https://github.com/aimclub/FEDOT/pull/1462) находится в нашей истории.
Он включает контракты подготовки TensorData, внешних моделей и transforms,
scoped-регистрацию extensions, полные исходы оценивания, retry/cache ownership
и точки подключения эволюционного оценивания. Не проектировать их повторно
как независимые механизмы, якобы отсутствующие в FEDOT.

[Industrial PR #318](https://github.com/aimclub/Fedot.Industrial/pull/318) закрепил
интеграцию именно с этим FEDOT SHA. В Industrial используются наш `create_data`,
`TensorData`, `Pipeline`, `PipelineNode`, `fedot.extensions` и TensorData-оценщик.
Следовательно, они уже существенно перешли на TensorData на внешней границе,
хотя внутри ряда implementations пока остаётся InputData.

`ForecastingTensorData`, `DetectionTensorData` и `TensorMultimodalBatch` Industrial —
обёртки над FEDOT TensorData и дополнительными доменными координатами/схемами,
а не конкурирующие замены общего контейнера. Их не нужно переносить целиком в FEDOT.

### Не обнаружена реализация, которая уже закрыла наши основные блокеры

- Native TensorData KFold/StratifiedKFold в проверенной интеграции не добавлены.
  У нас `DataSourceSplitter.build` всё ещё игнорирует `cv_folds`; их temporal-патч
  добавляет временной hold-out, а не общий CV.
- Native TensorData OOF не добавлен. Наш `PipelineOOFObjectiveEvaluate` всё ещё
  вызывает `tensordata_to_input_data`; это необходимо переделать независимо от Industrial.
- Полного переноса встроенных FEDOT моделей и data operations в Industrial PR нет.
  Их декларативный каталог описывает прежде всего Industrial-операции.
- Их нейросетевое ядро и compute prototype не заменяют обучение наших Torch,
  TabM/RTDL/RealMLP, sklearn/InterpretML и cuML моделей.

Эти выводы относятся к проверенному коду, не к будущим планам Ильи.

## 4. Где возможна существенная повторная работа

| Область | Что уже есть в Industrial | Риск для нашего полного перехода | Что учесть |
| --- | --- | --- | --- |
| Каталог и допустимость графа | Декларации model/transform, задачи, устройства, позиции узлов, число родителей, сериализация, structural roles; чистая проверка GraphSpec | Средний/высокий, если независимо построим второй несовместимый каталог и валидатор | FEDOT должен публиковать общие факты об операциях; Industrial — добавлять доменные ограничения |
| TS и меняющие строки transforms | Временной hold-out; `TemporalTransformInput/Output`; согласованное обновление окон, target и idx | Средний/высокий, если общий новый transform-контракт запретит любое изменение числа строк | Различать row-preserving и осмысленно row-changing операции; согласовать правила выравнивания |
| CV/OOF | Используется FEDOT evaluator; собственного native CV/OOF решения не обнаружено | Не дублирование алгоритма, но возможна поломка адаптера при смене producer | Оставить типизированные outcomes и версионирование folds; согласовать источник сведений о входе |
| Кеш и повторное использование | FEDOT cache identity patch; Industrial configuration identity, архив результатов и feature reuse | Средний, если смешать разные уровни identity или повторов | Разделить кеш признаков/предсказаний, результат CV и архив кандидатов |
| Исполнители | Явные sequential/process/Dask backend, бюджеты, владение ресурсами | Средний/высокий при одновременном создании параллельного runtime в FEDOT | Для первого CPU-search не требуется переписывать executor; перед параллелизмом согласовать границу callback/context |
| Модели и GPU | Независимые encoders/fusion; экспериментальный planner одной tensor operation | Низкий для уже сделанных адаптаций моделей | Заимствовать применимые контракты/тесты, а не заменять ими модельные реализации |
| Multimodal | Явная схема именованных модальностей, упаковка и восстановление, отдельные ветви графа | Средний на будущем image/multimodal этапе | Сохранить семантическую схему, а не закреплять flatten/packing как универсальный контейнер |
| `features` / `predict`, имена и импорты | Есть старые предположения и точные проверки классов стратегий | В основном локальный, хотя влияет на допустимость графов | Исправить адаптер и его тесты при синхронизации, не перестраивать ради него всё ядро |

### 4.1. Возможности операций: общий источник фактов, не общий гигантский planner

Industrial строит `IndustrialOperationDeclaration -> OperationCapabilities -> GraphSpec`.
Данные о backend, устройствах и форме исполнения проверяются до вызова алгоритмов.
Это пересекается с нашим предстоящим устранением legacy-узлов из поиска:
одной фильтрации node factory недостаточно, если mutation/initial assumption
могут напрямую добавить несовместимый узел.

Рекомендация: при проектировании полного FEDOT-пула использовать существующий
repository/extension metadata как общую границу, не вводить независимый второй
каталог с другими определениями model/transform/device. Доменные Industrial-поля
не обязательно все переносить в FEDOT. Проверка core не должна импортировать Industrial.

Полезно заранее согласовать: kind, входные/выходные data types и ограничения
ранга/формы, операции со строками/idx/target, позиции узлов, режимы исполнения
и сериализацию. Не обязательно реализовать всё сразу: добавлять подтверждённые
факты, необходимые текущему этапу, без ложных возможностей по умолчанию.

Особенно важно не путать:

- формат значения на границе extension (`numpy` / `torch`);
- устройство входного TensorData;
- устройство фактического вычисления алгоритма;
- устройство результата и явно разрешённые переносы.

Sklearn CPU adapter может принимать CUDA TensorData с явным переносом и возвращать
результат на CUDA, но от этого не становится GPU estimator. Копирование одного
поля `devices` без согласования его смысла даст неверный GPU-пул.

### 4.2. TS и row-changing transforms: важнее контракт, чем временный bridge

В temporal-патче есть полезные семантические требования:

- ряд делится по временной оси, а не по sample axis;
- отложенный горизонт не передаётся в обучение;
- train windows сопровождаются своими target и idx;
- inference формирует последнее окно без target;
- `forecast_length` согласуется с задачей;
- преобразование числа строк имеет явный контракт, обычный row-preserving transform
  не получает право произвольно менять строки.

Но текущая реализация `temporal_transform_request` использует
`legacy_temporal_input`, создаёт InputData и выполняет CPU NumPy-переход.
Поэтому **не переносить весь патч как окончательный TensorData-native runtime**.
Для нашей конечной цели сохранить семантику и независимые тесты, а окна и запрос
формировать непосредственно из TensorData. Для GPU отдельно проверить фактические
переносы и семантику backend, не ограничиваться типом возвращённого контейнера.

Патч поддерживает ограниченные сценарии; `validation_blocks > 1` отклоняется,
legacy-операции и создание forecasting-runtime ограничены одним одномерным рядом.
Это не завершённый multivariate TS/CV контракт.

При переносе `resample` и других row-changing операций решить ту же общую проблему:
после изменения строк должны быть согласованы target, idx и metadata. Не считать
специальный TS API автоматически подходящим для resampling.

### 4.3. CV/OOF: сохранить общий результат, не зависеть от устройства producer

Industrial `confirmed_input_from_objective` проверяет конкретный
`PipelineObjectiveEvaluateWithTensorData`, ожидает `_data_producer` типа `partial`
и читает `evaluator.tensor_data`. У нас это свойство сейчас возвращает первый
аргумент producer. При native CV producer может измениться, а OOF — использовать
другой evaluator: текущая интеграция может потерять сведения об исходном входе.

Это небольшой адаптерный риск, **не причина сохранять hold-out вместо CV**.
До смены producer согласовать явное описание входа для preflight-проверки,
не вычисляя folds повторно и не потребляя чужой одноразовый iterator.

Сохранять контракт `EvaluationComplete` / `EvaluationIncomplete`, `last_outcome`
и полное оценивание всех запрошенных folds. Fold identity должен учитывать
фактический split, seed и режим оценки. OOF и обычный CV не должны случайно
разделять кеш при разных правилах обучения промежуточных моделей.

### 4.4. Исполнители и кеш: не смешивать обязанности

Industrial executor принимает payload и callback оценки; его планы не являются
новым TensorData-контейнером или заменой Pipeline. Есть ограничение времени,
сериализация, spawn-процессы, Dask и явное владение клиентом.

Для native TensorData поиска сначала допустим последовательный backend.
При добавлении process/Dask согласовать перенос `extension_scope`, seed,
выбор устройства, lifecycle fitted state, кеш и отмену. Текущий RNG scope
Industrial фиксирует Python/NumPy; это не гарантия seed/determinism Torch/CUDA.
Process/Dask готовность на CPU не доказывает безопасный GPU-параллелизм.

Разделять уровни:

- FEDOT node cache / fold evaluation identity;
- наш chunked reuse с его областью данных и схемой подготовки;
- Industrial archive/configuration identity и локальный feature reuse.

Одинаковая структура графа ещё не означает одинаковую оценку: важны данные,
folds, seed, параметры, подготовка, device и extensions. Не переиспользовать
готовый результат Industrial как FEDOT CV outcome без согласования этих условий.

### 4.5. Модели, GPU и независимое нейросетевое ядро

`fedot_ind.neural` — нейтральные encoders, attention, fusion и tensor blocks.
Это не полный AutoML runtime и не замена наших training loops. Выделение
алгоритмических Torch-блоков из контейнерных адаптеров полезно и нам, но из этого
не следует зависимость FEDOT core от Industrial или перенос всех моделей туда.

Compute prototype расположен в `benchmark.industrial.experiments.compute_profile`,
а не является публичным общим GPU scheduler. Он исполняет одну tensor operation;
process/Dask для неё отклоняются. Не объявлять эту работу готовым GPU-поиском.

Наши доработки моделей продолжают быть нужны. TensorData-only не запрещает
NumPy/CuPy внутри явно определённой границы библиотеки, но не должно означать
скрытый переход всего графа в InputData. GPU search требует отдельно проверять
совместимость операций, размещение данных, память и число одновременно обучаемых моделей.

### 4.6. Multimodal и небольшие расхождения

Industrial уже хранит порядок и формы именованных модальностей и восстанавливает
их перед FUTURE. Для обхода ограничений текущего FEDOT creator 4D изображения
транспортируются через другое представление с сохранением исходной схемы.
При нашем расширении рангов нужно пересмотреть этот workaround, а не вводить
ещё один несовместимый порядок осей или безвозвратный flatten.

Разницу model-parent `predict` против `features` считать локальной синхронизацией
исполнения и валидатора. Наша конечная семантика — модель передаёт предсказание,
transform передаёт новые признаки; под старую предпосылку Industrial её не откатывать.
Точные сравнения sklearn-классов в их capability adapter заменить/обновить при
согласовании metadata и переименований. Эти различия сами по себе не требуют
массовой переделки FEDOT.

InputData, пока необходимый отдельным Industrial implementations, может временно
оставаться **в их адаптере**. Это не основание делать два постоянных пути для каждой
новой модели FEDOT. При окончательном удалении InputData API нужно синхронно
мигрировать зависящие от него Industrial wrappers/bridges или явно обновить
поддерживаемую пару версий; вечная совместимость с ними не является нашей целью.

## 5. Четыре подготовленных FEDOT-патча: что брать и чего не считать готовым

В PR #319 это файлы `fedot_ind/integration/fedot/compat/patches/`.
Они явно рассчитаны на `d1875e7a`; автоматического применения при импорте нет.

| Патч | Ценность для полного TensorData FEDOT | Результат read-only `git apply --check` на нашей рабочей копии |
| --- | --- | --- |
| `classification_tensor.patch` | Выход старых sklearn classifiers — Torch tensor на исходном device; regression tests | Накладывается; полноценный CUDA-input fit этим не реализован |
| `temporal_evaluation.patch` | TS split, horizon, window targets/idx, TensorData regression output | Накладывается, но содержит legacy bridge: это не конечная native реализация |
| `aligned_feature_fusion.patch` | Не терять связь индексов и строк на согласованных ветвях | Не накладывается автоматически из-за изменённого merger |
| `prediction_cache_identity.patch` | Передать descriptive_id из fit; не кешировать разные узлы под общим None | Не накладывается автоматически из-за изменённого Operation |

Это проверка применимости текста патча, **не тест поведения и не гарантия
совместимости всего набора**. Патчи не применялись. Конфликты контекста не означают,
что исправления уже реализованы у нас.

Проблема индексов отдельно воспроизведена на нашей ветке: объединение двух ветвей
с `idx=[9,2,7]` вернуло `[2,7,9]`, оставив строки признаков в прежнем порядке.
Исправление нужно адаптировать к нашему выбору model-parent predict, не копировать
без проверки. Тесты Industrial ограничены согласованными ветвями; они не доказывают
корректность произвольных несовпадающих индексов, OOF и источников данных.

При переносе патчей убрать зависимость результата от фиксированных checksum и
путей старой структуры, опубликовать core-исправления и согласовать новый FEDOT pin
в Industrial. Не применять их compatibility CLI к нашей dirty working copy.

## 6. Контрольные точки, когда возвращаться к этой записке

1. **До реализации совместимого пула и графовых ограничений.** Согласовать общую
   metadata-границу, backend/device смысл и row-changing контракт. Наши mutation,
   crossover, initial assumptions и hard-coded вставки должны соблюдать один пул.
2. **При переносе старых моделей и transforms.** Использовать подготовленные
   регрессионные сценарии; InputData-конверсии не выдавать за завершённую миграцию.
3. **До native CV/OOF.** Согласовать описание входа для Industrial adapter,
   типизированные outcomes, split identity и защиту от leakage в model -> model.
4. **До TS/multimodal этапа.** Зафиксировать оси sample/time/channel, окна,
   горизонт, target/idx и schema lifecycle; переиспользовать тестовые эталоны Industrial.
5. **До process/Dask/GPU поиска.** Согласовать executor ownership, перенос extensions,
   RNG, сериализацию, отмену, время и размещение данных. Не выводить готовность из
   одного успешного model.fit на CUDA.
6. **При синхронизации после переноса файлов.** Обновить Industrial imports и проверки
   конкретных классов на канонические пути FEDOT. Старые import aliases в FEDOT не сохраняем.
   Это сопутствующая, а не главная задача полного TensorData-перехода.
7. **Перед совместным merge/release.** Зафиксировать пару FEDOT/Industrial SHA и
   зависимости; обновить Industrial compatibility profile, CI и документацию.

### Минимальные совместные acceptance checks

- Настоящий search, а не возврат initial assumption: несколько поколений,
  все изменения графа и final fit/predict на TensorData, classification/regression.
- Не создавать InputData внутри этого native пути: проверять мосты и места
  конструирования, а не только тип публичного входа и выхода.
- Действительные KFold/StratifiedKFold и согласованное число folds в tuner/budget;
  отдельно native OOF без leakage и без bridge.
- Native transform -> native model и Industrial extension с array/Torch runtime
  -> model; регистрация не течёт между кандидатами, метки классов и TRAIN schema
  сохраняются. Legacy InputData wrappers проверять отдельно как временную
  совместимость, не засчитывать их в готовность native пути.
- Слияние несортированных idx и политика дубликатов; model -> model получает predict,
  а не исходные features; не смешиваются OOF и обычные train predictions.
- TS hold-out по time axis и lagged windows без доступа к будущим значениям;
  row-changing операция переносит target/idx вместе с признаками.
- Реальная CPU/CUDA матрица совместимых моделей; отдельно сериализация/контексты
  для process/Dask. Не считать пропуск GPU-теста успешным GPU результатом.
- Поздний результат после timeout не публикует fitted state или кеш; folds,
  chunk reuse и extension manifests не пересекаются по ошибочному cache key.

## 7. Источники для повторной синхронизации

Постоянные ссылки ниже привязаны к проверенному Industrial SHA. Перед новым
решением сравнить их с актуальным HEAD и заново проверить статус PR.

- [Industrial PR #319](https://github.com/aimclub/Fedot.Industrial/pull/319).
- [Интеграция TensorData](https://github.com/aimclub/Fedot.Industrial/blob/d631c7b56aac0602c09444eb57175fa52ead26db/fedot_ind/integration/fedot/tensor.py).
- [Temporal wrappers](https://github.com/aimclub/Fedot.Industrial/blob/d631c7b56aac0602c09444eb57175fa52ead26db/fedot_ind/integration/fedot/temporal_tensor.py).
- [Multimodal transport](https://github.com/aimclub/Fedot.Industrial/blob/d631c7b56aac0602c09444eb57175fa52ead26db/fedot_ind/integration/fedot/multimodal.py).
- [Каталог расширений](https://github.com/aimclub/Fedot.Industrial/blob/d631c7b56aac0602c09444eb57175fa52ead26db/fedot_ind/integration/fedot/extensions/catalog.json) и [declarations](https://github.com/aimclub/Fedot.Industrial/blob/d631c7b56aac0602c09444eb57175fa52ead26db/fedot_ind/integration/fedot/extensions/contracts.py).
- [Graph capabilities adapter](https://github.com/aimclub/Fedot.Industrial/blob/d631c7b56aac0602c09444eb57175fa52ead26db/fedot_ind/core/optimizer/graph_validation/adapters.py).
- [Связь с FEDOT evaluator](https://github.com/aimclub/Fedot.Industrial/blob/d631c7b56aac0602c09444eb57175fa52ead26db/fedot_ind/integration/fedot/evaluation.py).
- [Executor boundary](https://github.com/aimclub/Fedot.Industrial/blob/d631c7b56aac0602c09444eb57175fa52ead26db/fedot_ind/core/optimizer/evaluation/executors.py).
- [Compatibility patches и ограничения](https://github.com/aimclub/Fedot.Industrial/blob/d631c7b56aac0602c09444eb57175fa52ead26db/fedot_ind/integration/fedot/compat/README.md).
- [NN-01: текущие границы нейросетевого ядра](https://github.com/aimclub/Fedot.Industrial/blob/d631c7b56aac0602c09444eb57175fa52ead26db/docs/dev_guide/nn01_verification.md).
- [EVO-07: ограничения compute prototype](https://github.com/aimclub/Fedot.Industrial/blob/d631c7b56aac0602c09444eb57175fa52ead26db/docs/dev_guide/evo07_verification.md).

Обращать внимание на даты и исправления после ревью: некоторые ранние README
ещё упоминают удалённые import aliases или прежнее положение compute prototype.
При расхождении проверять код и актуальные verification notes, а не копировать
историческое описание как текущий контракт.

В смежных FEDOT PR [#1472](https://github.com/aimclub/FEDOT/pull/1472) и
[#1467](https://github.com/aimclub/FEDOT/pull/1467) есть работа по таймаутам,
начальным предположениям и portfolio. Перепроверить при изменении этих подсистем;
их наличие не означает готовность native TensorData CV/OOF.

**Сохранение в Git:** записка включена в коммит реструктуризации вместе с её картой.
Экспериментальные benchmark results и остальные локальные материалы туда не включены.
