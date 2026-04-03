# Architecture RFC: GPU trainer / hook / registry слой в `fedot/industrial`

## Кратко

Этот документ фиксирует текущее состояние нового слоя обучения нейронных сетей на GPU в `fedot/industrial`
и оценивает его с точки зрения текущей архитектурной линии проекта:

- `TensorData-first`
- `InputData` как compatibility shell
- сохранение OOP shell на внешней границе
- вынос planning / normalization / dispatch / compatibility logic в pure rules там, где это возможно

Основной scope анализа:

- `fedot/industrial/core/models/nn/utils/*`
- `fedot/industrial/core/models/nn/network_impl/base_nn_model.py`
- `fedot/industrial/core/models/nn/network_impl/llm_trainer.py`
- `fedot/industrial/tools/registry/*`
- текущие точки интеграции с `industrial strategies` и benchmark bridge в `fedot/core`

Главный вывод  -  направление решения сильное и стратегически правильное,
но текущая реализация пока выглядит как industrial-internal runtime shell с частично стабилизированным контрактом, 
а не как готовая общая training boundary для всего FEDOT. 
Следующий шаг -  выравнивании контракта hooks, отделении pure planning от effectful training shell 
и явном разведении data boundary и runtime metadata boundary.

## Зачем этот слой нужен

У нового слоя есть понятная и важная цель:

1. Перестать держать логику обучения внутри отдельных моделей в виде ad-hoc циклов.
2. Вынести policy-сценарии обучения, такие как early stopping, scheduler renewal, freezing, checkpointing и fit-reporting, в переиспользуемые hooks.
3. Добавить системный слой управления памятью и checkpoint lifecycle для GPU runtime.
4. Подготовить почву для сценария, где нейронные модели обучаются как отдельные узлы pipeline, а затем участвуют в population-level evolutionary evaluation, в том числе при ограниченной GPU памяти. 
5. Это хорошо совпадает с задачами текущей ветки TensorData-first roadmap.  

## Где этот слой находится в общей архитектуре

Сейчас слой можно мысленно разбить на четыре уровня.

### 1. Trainer interfaces

- `ITrainer`
- `IHookable`

Это тонкий верхний контракт, который говорит, что trainer умеет `fit(...)`, `predict(...)`, 
регистрировать дополнительные хуки и инициализировать циклы хуков.

### 2. Trainer shell

- `BaseTrainer`
- `BaseNeuralModel`
- `BaseNeuralForecaster`
- `LLMTrainer`

Это "слепкий", который имеют цикл обучения, `history`, `optimizer` / `scheduler`, тип девайса, registry cleanup 
и преобразованием предсказаний обратно в output container.

### 3. Hook layer

- `BaseHook`
- `HooksCollection`
- `LoggingHooks`
- `ModelLearningHooks`

Это policy-механизм, через который training shell пытается реализовать управляемое поведение по эпохам 
без жёсткого кодирования логики в каждой модели.

### 4. Registry layer

- `ModelRegistry`
- `CheckpointManager`
- `RegistryStorage`
- `MetricsTracker`

Это слой для загрузки чекпоинтов, cleanup GPU memory, хранения истории чекпоинтов и попытки привязать t
raining/runtime состояние к `fedcore_id` и `model_id`.

## Текущий integration path в FEDOT

Важно зафиксировать реальный, а не желаемый путь интеграции.

1. Базовые trainer-классы и registry живут внутри `fedot/industrial`.
2. Industrial neural models наследуются от `BaseNeuralModel` и используют этот слой как training runtime.
3. Industrial strategies подключают эти модели к industrial runtime FEDOT.
4. В `fedot/core` этот слой виден не напрямую, а через узкий адаптер:
   - `fedot/core/operations/evaluation/industrial_nn_bridge.py`
   - `fedot/core/operations/evaluation/industrial_nn_bridge_rules.py`
5. Следовательно, на текущем этапе это не общий FEDOT trainer, а industrial-internal abstraction с точечной интеграцией в core.
Это архитектурно важно. Документ не рекомендует объявлять этот слой уже сейчас новым универсальным 
training standard для всего `fedot/core`. Сначала нужно стабилизировать контракт и убрать внутренние противоречия.

## Иерархия и роли классов

### `ITrainer`

Ответственность:

- задаёт минимальный trainer contract через `fit(...)` и `predict(...)`

Входы/выходы:

- на вход принимает произвольный `input_data`
- возвращает trained object или prediction object

Владеемое состояние:

- не задаёт

Side effects:

- не описаны интерфейсом

Место в архитектуре:

- это формальный верхний контракт, но он пока слишком узкий, чтобы быть устойчивым

Оценка:

- плюс: интерфейс выделен явно
- минус: из него никак не следует логикак "наличия" циклов обучения, этапности хуков, требований к dataloader и проч

### `IHookable`

Ответственность:

- описывает объект, который умеет регистрировать дополнительные хуки и инициализировать циклы хуков

Входы/выходы:

- вход: iterable hooks
- выход: нет

Владеемое состояние:

- интерфейс не фиксирует, где именно hooks хранятся и как упорядочиваются

Side effects:

- изменение внутренней training scheme

Место в архитектуре:

- это правильная попытка выделить систему хуков как отдельный контракт, а не как скрытую деталь моделей

Оценка:

- плюс - hook layer назван и вынесен в интерфейс
- минусы - не определены ни стадии, ни в каком порядке "правильно" выполнять хуки , ни требования к trigger/action контракту

### `BaseTrainer`

Ответственность:

- общая "абстракция"" для цикла обучения
- хранит `params`, `learning_params`, `trainer_objects`, `history`, `model`, `device`
- даёт сервисные методы для cleanup, loss aggregation, kw normalization, field extraction и registry registration

Ключевые методы:

- `register_additional_hooks(...)`
- `_init_hooks()`
- `execute_hooks(...)`
- `_clear_cache()`
- `_compute_loss(...)`
- `_normalize_kwargs(...)`
- `_extract_output_fields(...)`
- `_register_model_checkpoint(...)`

Входы/выходы:

- вход: params-like dict
- выход: базовый shell для наследников

Владеемое состояние:

- `self.params`
- `self.learning_params`
- `self._hooks`
- `self._additional_hooks`
- `self.hooks_collection`
- `self.trainer_objects`
- `self.history`
- `self.model`
- `self.device`

Side effects:

- очистка CUDA cache
- обращение к `ModelRegistry`
- логирование
- регистрация checkpoints

Место в архитектуре:

- это правильный OOP shell-кандидат: именно он должен владеть "циклом обучения модели" и сопуствующими эффектами

Предварительная оценка:

- сильная сторона - централизация цикла обучения нейронок
- слабая сторона - данная реализация пока смешивает оркестрацию, частичное планирование и effectful runtime-поведение

### `BaseNeuralModel`

Ответственность:

- реализует операцию обучения/инфереса для нейронной модели на "уровне ноды в Pipeline"
- работает с dataloader-centric input contract
- управляет циклами хуков
- конвертирует prediction обратно в `TensorData`

Ключевая логика `fit(...)`:

1. Берёт `train_dataloader` и `val_dataloader` из входного объекта.
2. Определяет `task_type`.
3. Если `self.model` не задана, пытается взять модель из `input_data.target`.
4. Переносит модель на устройство.
5. Инициализирует hooks.
6. Запускает `_train_loop(...)`.
7. Делает cleanup cache.

Ключевая логика `_train_loop(...)`:

1. Прогоняет `train_loader` через `DataLoaderHandler.check_convert(...)`.
2. Для каждой эпохи запускает hooks начала эпохи.
3. Вызывает `_run_one_epoch(...)`.
4. Вызывает hooks конца эпохи, включая evaluation hooks.

Ключевая логика `predict(...)` / `_predict_model(...)`:

1. Делает device adjustment для quantized case.
2. Конвертирует `val_dataloader` через `DataLoaderHandler.check_convert(...)`.
3. Собирает batch predictions.
4. Периодически очищает память.
5. Прогоняет результат через `_convert_predict(...)`.
6. В `_convert_predict(...)` регистрирует checkpoint через registry и возвращает `TensorData`.

Входы/выходы:

- вход: объект с `train_dataloader` / `val_dataloader`, `task`, иногда `target`
- выход fit: обученная модель
- выход predict: `TensorData`

Владеемое состояние:

- `epochs`, `batch_size`, `learning_rate`
- `criterion`, `custom_criterions`
- `hooks`, `_hooks`, `_additional_hooks`
- `task_type`, `label_encoder`, `is_regression_task`
- `checkpoint_folder`, `batch_limit`, `calib_batch_limit`

Side effects:

- GPU training
- registry checkpoint registration
- cleanup cache
- batch-level tqdm logging

Место в архитектуре:

- это главный training shell для обычных нейронных моделей

Оценка:

- плюс - хороший кандидат на единый shell для node-level GPU training
- минус - входной контракт пока не выглядит архитектурно чистым, потому что trainer ожидает не столько `TensorData`, 
- сколько объект с dataloader-ами и иногда моделью в `target`

### `BaseNeuralForecaster`

Ответственность:

- специализация `BaseNeuralModel` под прогнозирование временных рядов
- добавляет логику in-sample / out-of-sample rollout
- владеет horizon-specific поведением

Ключевые поля:

- `train_horizon`
- `test_horizon`
- `in_sample_regime`
- `use_exog_features`
- `forecasting_blocks`
- `loss`
- `val_interval`

Ключевые методы:

- `out_of_sample_predict(...)`
- `create_features_from_predict(...)`
- `in_sample_predict(...)`
- forecasting-specific `_run_one_epoch(...)`
- forecasting-specific `_predict_model(...)`

Входы/выходы:

- вход: batch вида `(x_hist, x_fut, y)`
- выход: forecasting predictions

Владеемое состояние:

- forecasting-specific horizons and mode flags

Side effects:

- те же, что у `BaseNeuralModel`

Место в архитектуре:

- это хороший пример того, как "специализация под задачи" должна жить поверх общего training shell, а не через отдельный ad-hoc loop

Оценка:

- плюс - специализация отделена явно
- минус - forecasting path наследует все текущие внутренние проблемы базового shell, включая hook contract и registry side effects

### `BaseHook`

Ответственность:

- задаёт базовый hook contract через `trigger(...)` и `action(...)`
- инкапсулирует поведение, запускаемое на трениворочном цикле

Входы/выходы:

- вход: `epoch`, словарь kwargs
- выход: нет

Владеемое состояние:

- `params`
- `model`
- class-level `_SUMMON_KEY`
- class-level `_hook_place`

Side effects:

- зависят от конкретного hook: optimizer mutation, scheduler mutation, stop flag, checkpoint save, logging

Место в архитектуре:

- это policy mechanism, через который training shell должен делегировать сценарии обучения

Оценка:

- плюс - правильное концептуальное направление, hooks хорошо ложатся на сценарии freezing / early stop / saver / evaluator
- минус - текущий API hooks внутренне рассогласован и это уже blocker, а не просто технический долг

### `HooksCollection`

Ответственность:

- хранит hooks начала и конца эпохи
- сортирует hooks по приоритету
- управляет append/extend/clear

Входы/выходы:

- вход: экземпляры `BaseHook`
- выход: списки hook-ов

Владеемое состояние:

- `_on_epoch_start`
- `_on_epoch_end`

Side effects:

- мутация внутреннего порядка hooks

Место в архитектуре:

- это контейнер policy-layer, который должен быть тонким и предсказуемым

Оценка:

- плюс: идея правильная, отделён storage/order слой для hooks
- минус: текущая реализация не согласована с самими hook-классами и с кодом, который её использует

### `trainer_factory.create_trainer(...)`

Ответственность:

- выбрать trainer class по `task_type`, `params` и модели
- инкапсулировать решение `general` vs `forecasting` vs `llm`

Как работает сейчас:

1. Если `params['is_llm'] is True`, выбирается `LLMTrainer`.
2. Иначе анализируется имя класса модели и её `config`.
3. По набору строковых эвристик определяется `llm`, `forecasting` или `general`.
4. Для `forecasting` возвращается `BaseNeuralForecaster`.
5. Для остальных случаев возвращается `BaseNeuralModel`.

Входы/выходы:

- вход: `task_type`, `params`, `model`
- выход: trainer instance

Владеемое состояние:

- не хранит, это фабрика

Side effects:

- логирование

Место в архитектуре:

- это роутер между внешним task/model contract и trainer layer

Оценка:

- плюс - наличие единой точки выбора trainer - это уже полезно
- минус - "матчинг на строках"и не является устойчивым  решением для "типизированного сценария"

### `ModelRegistry`

Ответственность:

- быть coordination shell для циклов чекпоинтов модели, GPU cleanup, registry context и связи `fedcore_id <-> model_id <-> checkpoint`

Ключевые методы:

- `register_model(...)`
- `register_changes(...)`
- `update_metrics(...)`
- `save_metrics_from_evaluator(...)`
- `load_model_from_latest_checkpoint(...)`
- `get_checkpoint_path(...)`
- `cleanup_fedcore_instance(...)`
- `force_cleanup()`
- `set_registry_context(...)` / `get_registry_context()` / `clear_registry_context()`

Входы/выходы:

- вход: model object, `fedcore_id`, `model_id`, metrics, stage, mode
- выход: ids, checkpoint paths, loaded model, cleanup behavior

Владеемое состояние:

- singleton instance
- `CheckpointManager`
- `RegistryStorage`
- `MetricsTracker`
- thread-local registry context

Side effects:

- создание файлов checkpoint
- запись registry CSV
- очистка GPU memory
- удаление моделей из памяти

Место в архитектуре:

- это не просто хранилище, а рантайм объект для координации

Оценка:

- плюс - все гипотезы о "сохранении моделей и управлению памятью" действительно вынесены из моделей
- минус - singleton + thread-local context + сайд эффекты при работе с файловой системеой делают слой хрупким для parallel/Dask/evolutionary use cases

### `CheckpointManager`

Ответственность:

- сериализация модели в bytes
- сохранение чекпоинта на диск
- загрузка чекпоинта
- сбор GPU memory statistics
- очистка CUDA memory после save/load операций

Оценка:

- плюс: слой выделен отдельно и не смешан с registry storage
- минус: boundary между checkpoint bytes, model object и state_dict пока не стабилизирован как typed contract

### `RegistryStorage`

Ответственность:

- persistence layer поверх CSV/DataFrame
- append/update/get latest record/list ids

Оценка:

- плюс -  слой хранения отделён от оркестрации
- минус: CSV с полем `metrics` плохо подходит как долговременный типовой формат для сложных runtime сценариев

## Сильные стороны решения

### 1. Есть внятная попытка унифицировать training shell

Это самая сильная сторона решения. Вместо того чтобы каждая нейронная модель жила со своим закрытым training loop, 
проект движется к единому trainer boundary. Для дальнейшей интеграции в FEDOT это правильный вектор.

### 2. Hook-модель хорошо соответствует policy-style сценариям обучения

Freeze policy, early stopping, scheduler renewal, evaluation, logging и saver действительно удобно выражать через hooks. 
Это лучше, чем держать эти сценарии в условных ветках одного огромного `fit(...)`.

### 3. Memory/checkpoint concerns вынесены из моделей

`ModelRegistry`, `CheckpointManager` и `RegistryStorage` - это шаг в правильную сторону. 

Сам факт, что управление памятью и checkpoint lifecycle вынесено из нейронных моделей, стратегически важен для GPU runtime.

### 4. Dataloader-centric contract хорошо ложится на batched GPU training

Для нейронных сетей batch-based обучение и инференс естественен. 
В этом смысле слой уже думает в правильной плоскости и ближе к реальному GPU runtime, чем старые узлы, ожидающие только статические таблицы.

### 5. Архитектура разрабатывается с учетом population-level сценария

Даже если текущая реализация ещё не готова для надёжного parallel GPU execution, 
дизайн архитектуры сильный - checkpoint offloading и cleanup рассматриваются не как локальная деталь fit, 
а как часть большего эволюционного runtime.

## Слабые места и архитектурные риски

### 1. Planning и policy пока живут внутри effectful классов

Training policy, hook selection, optimizer/scheduler renew, memory cleanup  сейчас зашиты в effectful OOP shell. 
Это противоречит линии, по которой FEDOT старается выносить planning / normalization / dispatch в pure rules.

Практический вывод -  следующий шаг здесь должен быть не в "раздувании" shell, а в выделении pure planning helpers для:

- hook resolution
- optimizer/scheduler plan
- checkpoint policy
- cleanup policy
- trainer selection policy

### 2. Внутри слоя дублируется orchestration state

`BaseTrainer` держит `hooks_collection` как dict по стадиям, а `BaseNeuralModel` использует отдельный `HooksCollection`. 
Это означает, что в системе одновременно есть два способа моделировать один и тот же lifecycle state. 
Это недопустимо - shell уже не владеет своей orchestration model в одном месте.

### 3. Hook API внутренне несогласован

Это уже блокер. Наблюдаемые проблемы:

- в hook-классах используется `_hook_place`
- `HooksCollection` ожидает `HOOK_PLACE`
- `HooksCollection.start` и `HooksCollection.end` определены как методы
- `BaseNeuralModel._train_loop(...)` использует `self.hooks.start` и `self.hooks.end` как готовые списки, а не как методы

То есть текущий hook stack нарушает собственный контракт в нескольких местах сразу.

### 4. `TensorData` используется как carrier не только данных, но и runtime metadata

В `BaseNeuralModel._convert_predict(...)` и LLM path в output container прокидываются:

- `model`
- `checkpoint_path`
- `model_id`
- `fedcore_id`
 
В TensorData-first архитектуре `TensorData` должен быть первичной "внутренней моделью данных", 
но не контейнером для произвольного runtime ownership state.

### 5. Есть риск semantic overloading входных полей

`BaseNeuralModel.fit(...)` использует `input_data.target` как источник модели, если `self.model` ещё не задана. 
Это очень опасный сигнал. `target` в data container не должен неявно становиться transport-каналом для model object.

### 6. Registry слой слишком "неуправляемый" для parallel execution

Singleton + thread-local context + файловый persistence + eager cleanup - это сложное сочетание даже для обычного multi-threading, 
а для Dask/multi-process/multi-GPU scenario оно особенно рискованно.

Главные риски:

- race conditions при параллельной регистрации checkpoints
- неявное наследование context между задачами
- конфликт ownership у `fedcore_id` / `model_id`
- cleanup не того объекта в многозадачном runtime

### 7. `trainer_factory` пока не является typed boundary

Сейчас выбор trainer базируется на строковых эвристиках по имени класса и полям `config`.
Это допустимо как временный подход, но не как долгосрочный integration contract.

### 8. Почти нет mirrored tests, фиксирующих поведенческий контракт

Для слоя такого уровня это критично. Без тестов сейчас не зафиксированы:

- порядок hooks
- stop semantics
- scheduler renew semantics
- registry fallback behavior
- predict-time checkpoint behavior
- compatibility with TensorData/InputData boundaries
- single-GPU parallel safety invariants

## Конкретные code-level findings


1. Рассинхрон `HOOK_PLACE` vs `_hook_place`.
   В `hooks.py` hook-ы определяют `_hook_place`, а `HooksCollection` сортирует и маршрутизирует по `hook.HOOK_PLACE`. 
   В таком виде контракт уже сейчас выглядит сломанным.

2. `HooksCollection.start` / `HooksCollection.end` используются как атрибуты при том, что это методы.
   `BaseNeuralModel._train_loop(...)` и `FedCoreTransformersTrainer` итерируются по `self.hooks.start` и `self.hooks_collection.start` 
    без вызова. Это указывает на рассинхрон между API контейнера и его фактическим использованием.

3. Класс `Freezer` определён дважды.
   Это не только шум, но и риск тихого рассогласования, если версии класса начнут расходиться.

4. В `EarlyStopping` есть явные баги.
   В `action(...)` используется `self.count`, хотя поле называется `self.counts`. Значит stop flag в текущем виде ненадёжен.

5. `BaseTrainer._compute_loss(...)` может обращаться к `additional_losses` до инициализации.
   Если `model_output` имеет `.loss` или не является `torch.Tensor`, переменная `additional_losses` может не быть определена до `reduce(...)`.

6. В `HooksCollection.__init__(hooks=[])` используется mutable default.
   Это не фатально само по себе, но для orchestrator-контейнера это плохой сигнал.

7. В `MetricsTracker.find_best_checkpoint(...)` сортировка идёт по полю `version`, тогда как registry records сохраняют `created_at`.
   Это похоже на реальный дефект слоя registry.

8. В `LLMTrainer` свойство `epochs` объявлено дважды.
   Это показатель того, что слой ещё не стабилизирован как единый trainer contract.

9. Есть дублирование `hooks_collection` и `_additional_hooks` между `BaseTrainer`, `BaseNeuralModel` и `FedCoreTransformersTrainer`.
   Сейчас это выглядит как несколько частично пересекающихся механизмов расширения.

10. Predict path имеет скрытые side effects.
   `BaseNeuralModel._convert_predict(...)` и LLM path регистрируют checkpoint уже на предсказании. Это означает, что predict не является чистым inference boundary.

11. В registry storage поле `metrics` хранится в CSV/DataFrame и затем обновляется как dict-like структура.
   Для долговременной и переносимой persistence semantics это очень хрупко.

12. Есть признаки contract drift между `BaseNeuralModel` и унаследованными моделями.
   Например, такие модели, как `InceptionTimeModel` и `ResNetModel`, всё ещё содержат собственные `_init_model(...)` / `_fit_model(...)` / `_predict_model(...)` паттерны из более старого стиля, а `BaseNeuralModel.fit(...)` уже живёт по другому lifecycle.

## Насколько хорошо это интегрируется в текущую архитектуру FEDOT

### Что интегрируется хорошо

- идея отдельного training shell хорошо совпадает с OOP shell парадигмой
- hook layer как policy mechanism хорошо совпадает с будущим выносом planning в pure rules
- registry как отдельный subsystem полезен для GPU-aware runtime
- benchmark bridge уже даёт узкую точку входа из `fedot/core` без полного industrial monkey patch

### Что интегрируется слабо

- data/runtime boundaries ещё смешаны
- TensorData используется не только как data model
- trainer input contract опирается не на чистый `TensorData`, а на объект с dataloader-ами и местами overloaded semantics
- planner logic не вынесен из shell
- integration с `fedot/core` пока benchmark-only и точечный

## Интеграционная оценка по слоям

### `fit` как pipeline node

Оценка: средне.

Плюсы:

- есть общий training shell
- есть перспектива унифицировать fit нейронных узлов
- dataloader-centric runtime для GPU здесь уместен

Минусы:

- текущий fit boundary не прозрачен с точки зрения FEDOT data contracts
- часть старых моделей выглядит не до конца согласованной с новым fit lifecycle

### `predict` / `predict_for_fit`

Оценка: средне-низко.

Плюсы:

- prediction возвращается через единый output path
- есть единое место для post-processing

Минусы:

- predict path не является чистым inference boundary из-за checkpoint registration
- output container смешивает prediction data и runtime metadata
- `predict_for_fit` в основном повторяет `predict`, но не оформлен как отдельная ясно типизированная стадия

### Industrial strategy layer

Оценка: средне.

Плюсы:

- industrial strategies уже умеют использовать neural models как operation implementation
- integration остаётся локализованной внутри `fedot/industrial`

Минусы:

- trainer layer пока не выступает стабильным сервисным контрактом для strategy layer
- некоторые стратегии и модели выглядят как смесь старого и нового runtime-подхода

### Core bridge

Оценка: ограниченно хорошо.

Плюсы:

- benchmark bridge в `fedot/core` сделан узко и без возврата к глобальному industrial monkey patch
- это соответствует линии ветки: маленький vertical slice вместо broad rewrite

Минусы:

- bridge пока интегрирует слой как benchmark-only runtime adapter, а не как устойчивую архитектурную boundary
- проблемы внутри trainer/hook/registry слоя напрямую просачиваются в benchmark execution

### TensorData compatibility

Оценка: средне-низко.

Плюсы:

- `TensorData` уже используется как основной carrier prediction path
- это согласуется с общей линией TensorData-first

Минусы:

- `TensorData` заполняется runtime metadata, что размывает data boundary
- входной training contract всё ещё больше похож на dataloader wrapper, чем на чистый TensorData-first shell
- ownership между `TensorData`, model object и registry identifiers пока не зафиксирован

### Parallel population training on single GPU

Оценка -  концептуально перспективно, практически еще не готово.

Плюсы:

- сама постановка задачи правильная
- registry/cleanup/checkpointing уже думают в терминах memory pressure

Минусы:

- singleton registry и thread-local context слишком хрупки для безопасного параллелизма
- нет явной memory policy вида "один training worker на одно GPU-устройство"
- ownership checkpoint-ов и cleanup boundary пока не зафиксированы тестами

### Parallel population training on multi-GPU

Оценка - пока скорее гипотеза в сторону рисерча.

Плюсы:

- при правильной стабилизации trainer/registry слой может стать хорошей базой для multi-GPU orchestration

Минусы:

- нет typed device ownership policy
- нет явного scheduler contract для multi-device routing
- нет ясного isolation boundary между workers
- registry context в текущем виде не выглядит надёжным для multi-process сценариев

## Подробная логика работы методов и сценариев

### Сценарий 1. Обучение обычной нейронной модели как узла pipeline

1. В strategy layer создаётся concrete model class, наследующий `BaseNeuralModel`.
2. В `fit(...)` trainer получает объект с dataloader-ами.
3. Выбирается `task_type`, подхватывается `model`.
4. Выполняется `_init_hooks()`.
5. Стартует `_train_loop(...)`.
6. Hooks начала эпохи могут создать optimizer, scheduler и применить freeze policy.
7. `_run_one_epoch(...)` считает batch losses и пишет историю.
8. Hooks конца эпохи могут выполнить validation, early stopping, logging и saver behavior.
9. После fit вызывается cleanup cache.

### Сценарий 2. Предсказание и упаковка результата

1. `predict(...)` вызывает `_predict_model(...)`.
2. Данные читаются из dataloader.
3. Модель делает batched forward pass.
4. `_convert_predict(...)` нормализует prediction формат.
5. Затем происходит `_extract_output_fields(...)`.
6. Затем вызывается `_register_model_checkpoint(...)`.
7. В output container упаковываются и data fields, и runtime metadata.

Этот шаг особенно важен архитектурно. Именно здесь сейчас сильнее всего смешиваются data boundary и runtime boundary.

### Сценарий 3. Hook-driven training policy

1. Hooks собираются в `_init_hooks()` на основе params.
2. Для каждой эпохи `HooksCollection` должен отсортировать hooks по приоритету.
3. Hooks начала эпохи управляют optimizer/scheduler/freezing.
4. Hooks конца эпохи управляют validation, early stopping, saver и reporting.

Концептуально это сильное решение. Практически его надо стабилизировать, потому что сам contract hooks сейчас внутренне рассогласован.

### Сценарий 4. Registry-driven memory and checkpoint lifecycle

1. Trainer или predict path регистрирует модель через `ModelRegistry`.
2. `CheckpointManager` сериализует модель в bytes.
3. Checkpoint сохраняется на диск.
4. `RegistryStorage` добавляет запись в CSV-backed registry.
5. При необходимости `ModelRegistry` очищает модель из памяти и освобождает GPU cache.
6. Позже модель может быть восстановлена из latest checkpoint.

Концептуально это даёт путь к offloading heavy models при population-level evaluation.
Но именно здесь нужен самый аккуратный контракт ownership и cleanup policy.

## Decision-complete backlog

### P0. Выровнять hook contract и lifecycle invariants

Нужно сделать в первую очередь:

- унифицировать одно имя приоритета hook-а: либо `hook_place`, либо `HOOK_PLACE`, либо dataclass policy object
- унифицировать API `HooksCollection.start/end` как методы или как свойства, но не смешивать оба варианта
- зафиксировать trigger/action contract и типы `kws`
- убрать дубликат `Freezer`
- исправить `EarlyStopping`
- зафиксировать deterministic ordering hooks

Это P0, потому что без этого trainer layer не имеет стабильного внутреннего контракта.

### P1. Отделить pure planning/rules от training shell

Нужно вынести рядом с shell отдельные pure helpers/rules для:

- выбора hooks по params
- выбора optimizer/scheduler strategy
- построения cleanup plan
- checkpoint registration policy
- выбора trainer class

Это лучше всего продолжает текущую архитектурную линию FEDOT: OOP shell остаётся владельцем lifecycle, а decision logic уходит в pure core.

### P1. Зафиксировать ownership runtime metadata vs data container

Нужно явно решить, где должны жить:

- `model`
- `checkpoint_path`
- `model_id`
- `fedcore_id`

Рекомендация -  не расширять бесконтрольно `TensorData` как carrier runtime state. 
Лучше ввести отдельный runtime envelope или supplementary runtime record, а `TensorData` оставить главным образом data container.

### P1. Сделать explicit memory policy для one-GPU / multi-worker

Нужно явно зафиксировать правила:

- сколько trainer workers можно держать на одном GPU
- кто владеет cleanup
- когда допустим checkpoint offloading
- где проходит ownership boundary между worker, registry и evaluator

Без этой политики parallel GPU training останется непредсказуемым.

### P2. Унифицировать integration boundary между industrial trainer layer и core evaluation/runtime

Когда P0/P1 будут закрыты, тогда можно двигаться дальше:

- стабилизировать bridge contract для core
- убрать скрытые compatibility assumptions
- сделать trainer layer не benchmark-only bridge dependency, а более общий typed service boundary

### P2. Добавить mirrored tests и smoke scenarios

Сейчас слой остро нуждается в тестовом закреплении контракта. До этого любые последующие расширения будут хрупкими.

## Тестовый backlog

### Unit tests

- tests на порядок и trigger semantics hooks
- tests на `BaseTrainer._compute_loss(...)`
- tests на `_normalize_kwargs(...)`
- tests на `_extract_output_fields(...)`
- tests на registry fallback behavior при ошибках registry integration
- tests на `trainer_factory.create_trainer(...)` для classification / forecasting / llm cases

### Facade/service tests

- tests на `BaseNeuralModel.fit(...)`
- tests на `BaseNeuralModel.predict(...)`
- tests на `BaseNeuralForecaster` in-sample / out-of-sample regimes
- tests на predict-time contract без скрытого разрушения data boundary

### Integration tests

- tests через industrial strategies
- tests через core industrial bridge
- tests на `TensorData` compatibility round-trip для neural prediction path
- tests на checkpoint restore path

### Memory-policy tests

- tests с mocked CUDA stats
- tests на cleanup ownership
- tests на checkpoint offloading без потери предсказательного контракта

### Parallel-safety tests

- tests на thread-local registry context
- tests на ownership `fedcore_id/model_id`
- tests на отсутствие cross-talk между параллельными worker-ами
- tests на deterministic cleanup policy для single-GPU multi-worker scenario


## Итоговая оценка

Если смотреть стратегически, решение движется в правильную сторону. В проекте появляется:
1. единый shell для обучения нейронных сетей
2. hook-based policy layer 
3. отдельный registry subsystem для checkpoint/memory lifecycle. 
Это сильная база для будущего GPU-aware runtime в FEDOT.

Если смотреть инженерно на текущий код, слой пока ещё не достиг состояния стабильной архитектурной boundary.
Самые слабые места сейчас не в идее, а в контрактной реализации: 
1. hook lifecycle рассогласован
2. runtime metadata смешана с data container, 
3. registry не адаптирована для параллельного сценария,
4. часть моделей живёт на стыке старого и нового training подхода.

## Рекомендуемый следующий шаг:

1. стабилизировать hook contract,
2. вынести planning/rules из shell,
3. зафиксировать ownership data vs runtime metadata,
4. закрыть mirrored tests,
5. только после этого расширять слой как основу для parallel GPU training внутри evolutionary runtime.