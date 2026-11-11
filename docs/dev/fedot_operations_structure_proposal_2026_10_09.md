# Предложение по структуре операций FEDOT

Дата: 9 октября 2026 года; уточнение имён и происхождения классов — 30 октября.
Уточнение от 5 ноября: после согласования контейнерного контракта Linear/MLP и
TabM/RTDL/RealMLP объединяются в `NeuralTabular*Strategy`, модуль `strategies/models/neural.py`.
Дополнительно: сохраняется термин DataOperation; implementations группируются
по связанным семействам вместе с локальными базами, без правила «класс = файл».
Статус от 5 ноября: реструктуризация реализована по этому согласованному предложению.
Ниже сохранено обоснование и таблицы «до → после». Уточнение от 6 ноября:
прежние пути удалены, переэкспортов и старого дерева в актуальной ветке нет.
Точная карта всех модулей, фактические имена и оставшиеся шаги:
[результат реструктуризации](fedot_operations_restructuring_2026_11_05.md).

Текущее покрытие форматов данных описано отдельно:
[карта моделей и операций](fedot_operations_data_formats_2026_10_09.md).

## 1. Вывод

С замечанием о навигации и именах согласен. Проблема не в самом наличии стратегий,
а в смешении нескольких оснований группировки: задачи, библиотеки, исторического
происхождения кода и формата данных. Из имени и пути часто нельзя понять роль класса.

Сохранил бы три уровня: операция узла → стратегия исполнения → алгоритм/адаптер.
Разделил бы стратегии и реализации на соседние пакеты. Внутри стратегий сначала
отделил бы модели от data operations, затем группировал бы по библиотеке или общему
контракту вызова. Убрал бы исторические префиксы `Simple`, `Fedot` и постепенно `Tensor`.

При этом переезд файлов, переименование классов и объединение контрактов — три
разные задачи. Их не следует выполнять одним механическим массовым переименованием.

## 2. Что именно обнаружено

### 2.1. Общие интерфейсы смешаны с конкретными моделями

В `evaluation/evaluation_interfaces.py`
находятся не только `EvaluationStrategy`, но и `SkLearnEvaluationStrategy`, mapping
конкретных sklearn-моделей и функции multi-output адаптации.
Импорт базовой стратегии поэтому тянет конкретные sklearn-классы.

Это не чистый модуль интерфейсов. Базу стоит отделить от sklearn-интеграции.

### 2.2. Задача стала основанием для смешения разных ролей

В `evaluation/regression.py` находятся:

- `SkLearnRegressionStrategy`: вызов обычного sklearn estimator;
- `FedotRegressionStrategy`: вызов FEDOT model implementation, сейчас только KNN;
- `FedotRegressionPreprocessingStrategy`: преобразования данных, включая RFE,
  RANSAC, decomposition и isolation forest.

Уточнение: первая стратегия уже называется **SkLearnRegressionStrategy**, а не
просто `RegressionStrategy`. Но неоднозначность самого файла остаётся.

В `evaluation/classification.py`
аналогично смешаны обычный sklearn, FEDOT model wrappers и data operations для отдельных задач.
Кроме того, `FedotClassificationStrategy` объединяет sklearn wrappers LDA/QDA/SVC/KNN
и Keras CNN: ни `Fedot`, ни путь не объясняют это соседство.

В `evaluation/time_series.py`
рядом находятся forecasting-модели и преобразования временного ряда.

### 2.3. Названия отражают этап миграции, а не устойчивую роль

- `SimpleClassificationStrategy` / `SimpleRegressionStrategy` вызывают четыре
  базовые Torch-модели. `Simple` не задаёт контракт и ничего не говорит о библиотеке.
  Само имя `Torch` нормально: оно обозначает движок, а не обещает сложные архитектуры.
  Кроме того, MLP с активациями — нелинейная модель, не разновидность linear regression.
- `TensorSkLearn*Strategy` отличает новые модели от legacy sklearn по формату данных.
  Это полезно сейчас, но не является хорошим постоянным названием после миграции.
- `TensorTabular*Strategy` обслуживает TabM, FT-Transformer, ResNet и RealMLP.
  `Tensor` относится к внешнему контейнеру, а не гарантирует вычисления без NumPy.
  `Tabular` корректно описывает область применения этих моделей, но само по себе
  не объясняет отдельную стратегию: sklearn-модели тоже работают с таблицами.
  Для этой группы согласовано имя `NeuralTabular*Strategy`: оно описывает
  семейство нейросетевых табличных моделей, а не формат контейнера или движок.
- Каталог `gpu/` сейчас содержит cuML, а не все GPU-модели FEDOT. Torch-модели с CUDA
  находятся вне него. Поэтому `cuml` точнее, чем `gpu`.
- В именах встречаются `SkLearn` и `CuML`/`Cuml`. Для новых канонических имён
  выбрал бы единый стиль `Sklearn...` и `CuML...`.

### 2.4. Реализации вложены в пакет стратегий

`evaluation/operation_implementations/` логически отдельный слой, но физически
выглядит как внутренность evaluation. При этом
`evaluation/abstract_node.py`
содержит `TensorDataOperationImplementation`, а не абстрактный узел.

В новом `evaluation/tensor_sklearn.py`
EBM adapters тоже находятся рядом со стратегиями. В предложенной структуре им
место в реализациях, а не в стратегии.

## 3. Какие разделения действительно нужны

### Операция, стратегия и реализация

```text
PipelineNode
  → Operation / Model / DataOperation
      → выбранная по metadata стратегия
          → внешний estimator или FEDOT implementation
```

Операция обеспечивает интеграцию с графом, параметры, выбор стратегии и кеширование.
Стратегия согласует вызовы с конкретным семейством: как создать estimator, передать
данные, выбрать labels/probabilities, оформить результат и выполнить fit-stage вызов.
Реализация содержит алгоритм или его адаптацию: архитектуру, training loop,
специальную подготовку признаков, коррекцию KNN-параметров и т. п.

Поэтому отделить стратегии от реализаций — да. Удалить все стратегии и перенести
всё в `Operation` — нет: он станет знать особенности каждой библиотеки.
Но стратегия нужна на общий контракт, а не обязательно на каждый алгоритм.

### Термин DataOperation сохраняется

Переименование общей категории в Transform не требуется. `DataOperation`,
`DataOperationImplementation`, ключ репозитория `data_operation` и каталог
`data_operation_repository.json` уже связаны с фабрикой, metadata и графом.
У этого обозначения нет проблемы, которую решало бы массовое переименование.

Поэтому в новом дереве используются `strategies/data_operations/` и
`implementations/data_operations/`, а в предлагаемых именах классов —
`RegressionDataOperationStrategy`, `ClassificationDataOperationStrategy` и т. п.
Методы `transform` / `transform_for_fit` сохраняются: это название действия,
а не требование переименовать категорию операции.

Существующий внешний API `ExternalTransformSpec`, `TransformCapabilities`,
`ExtensionTransformStrategy`, а также `TransformImplementation`-протокол в
`fedot.extensions.runtime_contracts` не переименовывается заодно. Это уже отдельный
array-level контракт внешних операций. Не смешивать его с core
`DataOperationImplementation` и не заводить второй core-класс с тем же именем
`TransformImplementation`. Изменение API extensions, если понадобится, требует
отдельной синхронизации с Industrial, а не механической замены слова Transform.

### Прямой estimator и FEDOT wrapper

У обычного sklearn estimator вызовы выглядят как `fit(X, y)` / `predict(X)`.
У `FedotKnnRegImplementation` — `fit(data)` / `predict(data)`; внутри ещё корректируется
`n_neighbors`, и implementation хранит исправленные параметры для узла.

Это реальное различие, объясняющее отдельный путь. Однако название
`FedotRegressionStrategy` его не выражает. Не стал бы переименовывать его в
`KNNRegressionStrategy`: код стратегии не реализует KNN и может обслуживать другие
модели с таким же контрактом.

### Classification, regression и data operations

Разделение classification/regression имеет смысл из-за probabilities, labels,
classes и multi-output поведения. Их можно держать в одном файле семейства.
Data operations нужно отделить от моделей: они изменяют признаки, могут использовать
target при обучении и иметь отдельный `transform_for_fit`.

Единое правило размещения: **разные классы по задаче не требуют разных файлов**.
Парные classification/regression стратегии одного семейства остаются в общем
модуле: `models/sklearn.py`, `models/neural.py`, `models/cuml.py` и аналогично
`data_operations/preprocessing.py`. Не разделять только одну пару на `regression.py`
и `classification.py`, оставляя остальные пары сгруппированными без причины.

### Базовые Torch-модели и нейросетевые табличные модели

У `evaluation/operation_implementations/models/torch.py`
`predict` / `predict_proba` возвращают сырой `torch.Tensor`, который оформляет стратегия.
У `evaluation/operation_implementations/models/tensor_tabular.py`
возвращается `TensorData`; методы и обработка вероятностей также не полностью совпадают.

Проверка кода подтверждает: **сами `Simple*Strategy.predict()` уже возвращают
TensorData**, как и `TensorTabular*Strategy`. Отличается нижняя граница вызова
implementations: упаковка результата, имя метода меток и форма вероятностей.
Это не причина сохранять отдельные стратегии для небольших Linear/MLP моделей.

Согласованное решение: привести implementations к общему контракту и объединить
группы в `NeuralTabularStrategy`, `NeuralTabularClassificationStrategy` и
`NeuralTabularRegressionStrategy`, модуль `strategies/models/neural.py`.
Отдельные `Simple*Strategy` и предложенные ранее `Torch*Strategy` не сохраняются.
Классификация и регрессия остаются разными стратегиями с общей базой fit.

В группе будут Linear, MLP, TabM, FT-Transformer, ResNet и RealMLP. Название обозначает
семейство табличных моделей с общей интеграцией, не строгую классификацию алгоритмов:
например, Torch linear regressor сейчас использует least squares.
Training loops, scaling, архитектуры и специальный preprocessing остаются внутри
implementations. Объединять их обучающие базы и алгоритмы ради общей стратегии не нужно.

### Согласованный контракт стратегий и нейросетевых implementations

Работа Operation со стратегиями должна оставаться универсальной. Формат результата
на границе стратегии не зависит от конкретной модели. В native TensorData-пути
стратегия возвращает TensorData; существующий legacy predict-контракт — OutputData
для входного InputData, а не возврат InputData. Это не требование добавить legacy-путь
уже перенесённым моделям.

| Уровень | Метод | Результат |
| --- | --- | --- |
| Стратегия | `fit(train_data)` | Обученный implementation / estimator, **не** контейнер данных |
| Стратегия | `predict(trained_operation, predict_data)` | TensorData с модельным результатом в `predict` |
| Стратегия | `predict_for_fit(trained_operation, predict_data)` | Тот же контейнерный контракт; отдельное поведение только при содержательной необходимости |
| Implementation объединённой группы | `fit(data)` | Обученный implementation (`self`) |
| Implementation классификатора | `predict(data)` | TensorData с метками классов в `predict` |
| Implementation классификатора | `predict_proba(data)` | TensorData с полной матрицей вероятностей `(n_samples, n_classes)` в порядке `classes_` |
| Implementation регрессора | `predict(data)` | TensorData с регрессионными предсказаниями в `predict` |

Численные внутренние методы могут возвращать сырые тензоры/массивы. Но делать
сырой тензор обязательным результатом публичного predict-контракта всех этих
implementations не нужно: сохраняем контейнерный подход уже имеющихся TabM/RTDL/RealMLP.
Внешние sklearn/cuML estimators не обязаны знать TensorData — упаковка на их
границе остаётся обязанностью стратегии/адаптера. Унификация не означает
переписывание методов сторонних библиотек под наш контейнер.

Необходимые изменения перед объединением:

1. У Linear/MLP implementations оформить `predict` и `predict_proba` в TensorData.
   Для классификации `predict` должен означать **метки**, а не вероятности, как
   сейчас унаследовано от `TorchTabularModel`. Согласовать использование
   `predict_labels`: общий dispatcher не должен выбирать его только для Simple.
2. У TabM/RTDL убрать внутреннее сворачивание бинарных вероятностей до одного
   столбца; полный `predict_proba` должен быть одинаковым у всех implementations.
   RealMLP уже возвращает полную матрицу. Само значение вероятностей не менять.
3. В общей классификационной стратегии оставить выбор `output_mode`: `labels`
   вызывает `predict`; `full_probs` возвращает всю матрицу; бинарные `probs/default`
   выбирают вероятность второго класса, multiclass сохраняет матрицу.
   `False` согласовать с `default`, как допускает общий validator режимов.
4. Возвращать новый TensorData с исходными `features`, `target`, `idx`, task и
   корректной metadata. Prediction должен быть Torch tensor на устройстве входа;
   входной контейнер не мутируется. Не переупаковывать уже оформленный TensorData
   как значение его поля `predict` и не рассчитывать на fallback в Operation.
5. Добавить Linear/MLP в mappings общей группы, обновить JSON strategies и убрать
   отдельные Simple-dispatchers. Operation ID `torch_linear`, `torch_mlp` и их
   regression-варианты сохранить. Не вводить ветки по типу конкретной модели
   вместо согласования интерфейса в implementations.

При общем fit отдельно сохранить поведение random state. Сейчас Simple не использует
`ImplementationRandomStateHandler`, а TensorTabular использует. У Torch-модели
`random_state` появляется внутри fit; слепое добавление handler может восстановить
в это новое поле `None` после обучения. Согласовать владение seed и fitted state,
а не менять его неявно ради общей обёртки.

Проверки при реализации: прямые вызовы implementations и стратегий; binary и
multiclass modes; regression shape; предсказание без target; порядок `classes_`;
CPU/CUDA и metadata; отсутствие мутации входа; повторный predict и сохранение/загрузка;
seed и параметры обучения. Обязательно сохранить model -> model передачу именно
`predict`, а не исходных признаков. Обновить существующие прямые тесты Torch-моделей,
которые сейчас ожидают сырой Tensor. OOF не подменять обычным train predict.

Это отдельное согласованное изменение внутреннего контракта с тестами, а не
механический перенос файлов. Настройки обучения, качество и preprocessing defaults
не должны меняться вместе с ним. Для остальных стратегий придерживаться того же
общего контейнерного контракта, если нет предметной причины для отличия.

## 4. Рекомендуемая целевая структура

Ниже целевой вариант, а не требование сразу создать все эти файлы.
Связанные implementations остаются группами в одном модуле, как сейчас
`sklearn_selectors.py`, `sklearn_filters.py` и `boostings_implementations.py`.
Пакет нужен только при реальной необходимости разделить самостоятельные подсистемы,
не потому, что моделей несколько. Пустые wrappers для внешних estimators не нужны.

```text
fedot/core/operations/
├── operation.py, model.py, data_operation.py, automl.py
├── factory.py, operation_parameters.py, ...
├── strategies/
│   ├── base.py                     # EvaluationStrategy — имя сохраняется
│   ├── models/
│   │   ├── sklearn.py              # все прямые sklearn-compatible estimators, не только уже migrated
│   │   ├── adapters.py             # FEDOT model implementation contract
│   │   ├── neural.py               # NeuralTabular*: Linear / MLP / TabM / RTDL / RealMLP
│   │   ├── cuml.py                 # cuML, DLPack, precision/device handling
│   │   ├── boosting.py
│   │   ├── foundation.py           # TabPFN / TabICL
│   │   ├── forecasting.py
│   │   └── custom.py
│   ├── data_operations/
│   │   ├── common_preprocessing.py # scaling, imputation, encoders, ...
│   │   ├── preprocessing.py        # RegressionDataOperationStrategy / ClassificationDataOperationStrategy
│   │   ├── projection.py           # PCA / SVD / KernelPCA / ICA / DaskPCA после адаптации
│   │   ├── time_series.py
│   │   ├── text.py
│   │   ├── optional.py             # мост к OptionalService
│   │   └── optional_config.py      # нынешний builder, не алгоритм
│   ├── sources.py
│   ├── automl.py
│   └── extensions.py
└── implementations/
    ├── base.py                     # общие ModelImplementation / DataOperationImplementation
    ├── models/
    │   ├── estimator_adapters.py   # EBM / KNN / SVC / LDA / QDA и другие нужные estimator adapters
    │   ├── neural.py               # Linear / MLP / TabM / RTDL / RealMLP и локальные базы
    │   ├── boosting.py             # XGBoost / LightGBM / CatBoost и локальные базы
    │   ├── foundation.py           # TabPFN / TabICL и общие bases/helpers
    │   ├── forecasting/            # существующие группы: statsmodels.py, arima.py, naive.py, ...
    │   ├── keras.py
    │   └── custom.py
    └── data_operations/
        ├── features_reducing.py    # PCA / SVD / KernelPCA / ICA / DaskPCA и семейные базы
        ├── sklearn_transformations.py # scaling / normalization / imputation / polynomial features
        ├── categorical_encoders.py # OneHot / LabelEncoding вместе
        ├── sklearn_selectors.py   # base + linear/nonlinear RFE для regression/classification
        ├── sklearn_filters.py     # base + RANSAC / IsolationForest варианты
        ├── decompose.py           # base + regression/classification decomposition
        ├── resampling.py
        ├── ts_transformations.py
        ├── text_preprocessing.py, text_pretrained.py
        └── topological/
```

Нельзя буквально применять одно основание группировки ко всем уровням:
стратегии группируются по общему исполнению; реализации — по связанному семейству,
общим базам и зависимостям. Стратегия/пара стратегий — хороший ориентир для общего
implementation-модуля, но не обязательное соответствие один к одному.
Не предлагаю создавать параллельные каталоги `cpu/`, `gpu/`, `tensor/`, `input/`.

Например, `NeuralTabular*Strategy` и `implementations/models/neural.py` содержат
общую группу моделей; classification/regression варианты одной модели не разделяются.
Но разные зависимости или самостоятельный алгоритмический контур могут оправдать
несколько модулей у одной стратегии. Уже сейчас FedotClassificationStrategy
использует `knn.py`, `svc.py`, `discriminant_analysis.py` и `keras.py` — строгого
правила «одна стратегия = один файл» в старом коде не было.

### Группировка implementations и размещение баз

- Общие контракты `ModelImplementation` и `DataOperationImplementation` находятся
  в `implementations/base.py`. Это общие границы вызова, не общий training framework.
- Базы и helpers, используемые только одним семейством, остаются в его модуле:
  например, обучающие базы Linear/MLP и базы TabM/RTDL — в `models/neural.py`,
  RFE base — в `data_operations/sklearn_selectors.py`. `EncodedInvariantImplementation`
  сейчас используется только в `sklearn_transformations.py`: если эта база останется
  нужна после TensorData-миграции, её место рядом с этими реализациями, не среди
  общих контрактов. Общая стратегия не требует
  общего предка для всех training loops.
- Базу выносим отдельно только при реальном использовании несколькими модулями
  или для устранения циклических зависимостей; не создавать `base.py` на каждый алгоритм.
- Существующие связанные группы по умолчанию сохраняем. Не разбивать selectors,
  filters, transformations, decomposition и пары classification/regression на файл
  для каждого класса. Не сливать все data operations в один файл только потому,
  что часть из них вызывается одной preprocessing-стратегией.
- `models/neural.py` в дереве — исходный вариант группировки, а не запрет на дальнейшее
  обоснованное разделение. Если размер, зависимости или развитие семейств действительно
  потребуют пакета, делить по связанным подсистемам, а не по одной модели/классу.
- Heavy optional dependencies загружать при использовании соответствующей модели.
  Группировка с TabM/RealMLP не должна делать их обязательными для Linear/MLP.

### Файлы рассчитаны на конечный каталог, не на этап TensorData-миграции

Размещение определяется назначением и контрактом, а не тем, какая операция первой
перенесена на TensorData. Последующая адаптация legacy-класса должна приводить его
в общую группу, а не создавать постоянную пару «старые» / «новые» модули.
При этом не нужно выдумывать будущие модели ради заполнения файла.

| Целевая группа | Что относится к ней после переноса | Что важно не закреплять в структуре |
| --- | --- | --- |
| `strategies/models/sklearn.py` | Прямые sklearn-compatible estimators из обоих нынешних mappings: RF, Logit, Ridge, HistGB, ExtraTrees, MLP и EBM через адаптер | Не оставлять отдельную стратегию только для первых шести migrated-моделей |
| `implementations/models/estimator_adapters.py` | Локальные адаптации EBM, KNN, SVC, LDA/QDA и другие действительно необходимые небольшие adapters внешних estimators | Не выделять маленький `interpret.py` только из-за отдельной библиотеки и текущей пары EBM |
| `implementations/models/neural.py` | Linear/MLP и TabM/RTDL/RealMLP вместе с локальными базами; последующие модели с тем же назначением | Не делить на native/legacy, TensorData/InputData или «простые»/«сложные» файлы |
| `strategies/data_operations/projection.py` и `implementations/data_operations/features_reducing.py` | PCA/SVD и адаптированные KernelPCA, FastICA, DaskPCA вместе с нужной component-analysis базой | Не оставлять группу проекций навсегда равной только уже перенесённым PCA/SVD |
| Остальные связанные группы | Boosting, foundation, forecasting, selectors, filters и т. д. по назначению | Не создавать пустые модули под гипотетическое расширение и не плодить wrappers ради единой раскладки |

`estimator_adapters.py` — адаптации внешних estimators к требованиям FEDOT,
не библиотека собственных реализаций всех sklearn-алгоритмов. Большинство RF,
Logit, Ridge и подобных моделей уже реализовано внешней библиотекой: после миграции
они остаются прямыми ссылками в strategy mapping, без пустого wrapper-класса на
каждую модель. EBM здесь sklearn-compatible по способу вызова, но принадлежит
InterpretML; нейтральное имя модуля не приписывает его пакету sklearn.

У EBM classifier adapter есть содержательная работа с числовыми метками, а
regressor adapter сейчас в основном обеспечивает lazy import. Оба можно держать
рядом с остальными небольшими estimator adapters. Импорты InterpretML оставлять
ленивыми: KNN/SVC/LDA не должны требовать установленный InterpretML.
Группировка в одном модуле сама по себе не требует одинакового наследования:
границу array-level estimator и FEDOT container-level implementation пока
различают соответствующие стратегии, а не имя файла.

При переносе старых проекций перемещать их component-analysis базу с потребителями
из `sklearn_transformations.py` в `features_reducing.py`; не разносить её наследников.
У общей preprocessing-стратегии после этого остаются общие преобразования, а
проекции попадают в общее projection-семейство. Это целевое изменение mappings
после адаптации, не утверждение о том, что текущие KernelPCA/ICA уже TensorData-ready.
Отдельные алгоритмические backend и optional dependencies при этом сохраняются.
Импорт DaskML должен быть ленивым и не требоваться для обычных Torch PCA/SVD.

Малое число классов само по себе не запрещает отдельный файл: например, custom
callables и Keras CNN имеют самостоятельный контракт/зависимости. Но отдельный
модуль должен объясняться такой границей, а не фактом переноса первой пары моделей.
Размер и состав файлов можно уточнять при полной инвентаризации перед реализацией.

`adapters.py` означает FEDOT implementations, которым передаётся контейнер данных,
в отличие от прямых sklearn estimators. Если после унификации контрактов окажется,
что отдельной стратегии там не нужно, её следует объединить, а не сохранять ради дерева.
Для Keras CNN отдельная стратегия нужна только при реальных особенностях вызова;
одна лишь другая библиотека не требует дополнительного класса.

Основные preprocessing services остаются в `fedot/preprocessing`: здесь находятся
только операции/адаптеры узлов графа, не второй независимый preprocessing framework.
Extensions сохраняют регистрацию и runtime в `fedot/extensions`.

## 5. Конкретные предлагаемые имена

`EvaluationStrategy` **не переименовываем**. Предлагается только вынести базовый
класс в `strategies/base.py`, отделив его от конкретной sklearn-интеграции.
Публичный доступ к стратегии через node/operation в это предложение не добавляется.

Таблицы разделены по роли, чтобы не смешивать стратегию вызова с самим алгоритмом.
Примеры моделей/операций — напоминания, не полные mappings. Имена вроде `rf`,
`torch_mlp` и `pca` — текущие operation ID; их переименование не предлагается.
В колонках «Где сейчас (до переноса)» и «Куда планируется» все пути указаны
относительно `fedot/core/operations/`. Исторические пути указаны текстом, поскольку старые файлы удалены;
планируемые — согласованные целевые пути из пункта 4. После реализации они являются
каноническими; колонка «Сейчас» сохраняет состояние до реструктуризации, включая
подготовительные изменения из `cd281e48`, а не только последние пять коммитов.

### Как читать колонку «В твоих пяти коммитах»

Сравнивается база `d1875e7a` с пятью коммитами до `04094a67` включительно:

- **М1** — `07e10fb6`: новые sklearn/InterpretML и TabM/RTDL/RealMLP модели;
- **М2** — `384d33d6`: OOF cross-validation;
- **М3** — `7aa33d17`: повторное использование вычислений при chunked search;
- **М4** — `e9369973`: native Torch-модели;
- **М5** — `04094a67`: восстановление cuML-интеграции.

Обозначения:

- **Новая** — класс впервые появился в этих пяти коммитах.
- **Была; доработана** — класс существовал до них, но его реализация менялась в них.
- **Была** — класс уже был в базе; в этих пяти коммитах сам не менялся.
- **После пяти** — класс выделен при наших последующих незакоммиченных изменениях.

Это происхождение относительно конкретного стека коммитов, а не утверждение об
авторстве всей истории. «Была» не означает, что ты никогда раньше над ней не работал.
М2/М3 относятся к оцениванию и поиску, а не к новым классам в этих таблицах.

### 5.1. Базовые стратегии

Это стратегии/базы стратегий, **не реализации моделей**.

| Сейчас | Роль | Примеры обслуживаемых моделей | Предлагаемое имя | Где сейчас (до переноса) | Куда планируется | В твоих пяти коммитах |
| --- | --- | --- | --- | --- | --- | --- |
| `EvaluationStrategy` | Общая база моделей и data operations | Все семейства через наследников | Имя сохранить | `evaluation/evaluation_interfaces.py` | `strategies/base.py` | Была; доработана · М1 |
| `SkLearnEvaluationStrategy` | База прямых sklearn estimators | RF, LogisticRegression, Ridge | `SklearnModelStrategy` | `evaluation/evaluation_interfaces.py` | `strategies/models/sklearn.py` | Была; доработана · М1 |
| `TensorSkLearnEvaluationStrategy` | База TensorData-only sklearn-compatible estimators | ExtraTrees, HistGradientBoosting, EBM | Целевой `SklearnModelStrategy` после миграции legacy | `evaluation/tensor_sklearn.py` | `strategies/models/sklearn.py` | После пяти; сами модели добавлены в М1 |
| `TensorTabularStrategy` | База вызова нейросетевых implementations | Сейчас TabM/RTDL/RealMLP; после унификации также Linear/MLP | `NeuralTabularStrategy` | `evaluation/tensor_models.py` | `strategies/models/neural.py` | Новая · М1 |
| `CuMLEvaluationStrategy` | База cuML estimators | cuML RF, Logit, Ridge, KMeans | `CuMLModelStrategy` | `evaluation/gpu/common.py` | `strategies/models/cuml.py` | Была; переработана · М5 |

В М1 в старую sklearn-базу добавили шесть моделей и нормализацию target.
Уже после пяти коммитов эти шесть моделей вынесены в `TensorSkLearn*`, чтобы
перевести только их на TensorData-only. Не нужно приписывать появление этих
отдельных классов стратегий самому М1.

Текущее имя `TensorSkLearnEvaluationStrategy` пока сохраняем. Удаление префикса
`Tensor` и объединение sklearn-баз относятся к будущему завершению миграции,
а не к промежуточному переименованию нейросетевых стратегий.

### 5.2. Стратегии моделей

Каждая строка — **стратегия**, а модели перечислены в соседней колонке.

| Сейчас | Вид стратегии | Примеры моделей / ID | Предлагаемое имя | Где сейчас (до переноса) | Куда планируется | В твоих пяти коммитах |
| --- | --- | --- | --- | --- | --- | --- |
| `SkLearnClassificationStrategy` | Классификация · прямой sklearn | `rf`, `logit`, `dt`, `mlp` | `SklearnClassificationStrategy` | `evaluation/classification.py` | `strategies/models/sklearn.py` | Была; менялась её база · М1 |
| `SkLearnRegressionStrategy` | Регрессия · прямой sklearn | `ridge`, `linear`, `rfr`, `gbr` | `SklearnRegressionStrategy` | `evaluation/regression.py` | `strategies/models/sklearn.py` | Была; менялась её база · М1 |
| `SkLearnClusteringStrategy` | Кластеризация · sklearn | `kmeans` | `SklearnClusteringStrategy` | `evaluation/clustering.py` | `strategies/models/sklearn.py` | Была |
| `TensorSkLearnClassificationStrategy` | Классификация · TensorData-only, CPU estimator | `extra_trees`, `hist_gb`, `ebm` | Целевой `SklearnClassificationStrategy` | `evaluation/tensor_sklearn.py` | `strategies/models/sklearn.py` | После пяти; модели из М1 |
| `TensorSkLearnRegressionStrategy` | Регрессия · TensorData-only, CPU estimator | `hist_gbreg`, `ebmreg`, `mlpreg` | Целевой `SklearnRegressionStrategy` | `evaluation/tensor_sklearn.py` | `strategies/models/sklearn.py` | После пяти; модели из М1 |
| `FedotClassificationStrategy` | Классификация · FEDOT wrappers | LDA, QDA, SVC, KNN; также Keras CNN | `AdapterClassificationStrategy` | `evaluation/classification.py` | `strategies/models/adapters.py` | Была |
| `FedotRegressionStrategy` | Регрессия · FEDOT wrapper | Только `knnreg` | `AdapterRegressionStrategy` | `evaluation/regression.py` | `strategies/models/adapters.py` | Была |
| `SimpleClassificationStrategy` | Классификация · native Torch | `torch_linear`, `torch_mlp` | Объединить с `NeuralTabularClassificationStrategy` после согласования implementations | `evaluation/simple_classification.py` | `strategies/models/neural.py` | Была; доработана, добавлен MLP · М4 |
| `SimpleRegressionStrategy` | Регрессия · native Torch | `torch_linear_reg`, `torch_mlp_reg` | Объединить с `NeuralTabularRegressionStrategy` после согласования implementations | `evaluation/simple_regression.py` | `strategies/models/neural.py` | Новая · М4 |
| `TensorTabularClassificationStrategy` | Классификация · нейросетевые implementations | `tabm`, `ft_transformer`, `tab_resnet`, `realmlp` | `NeuralTabularClassificationStrategy` | `evaluation/tensor_models.py` | `strategies/models/neural.py` | Новая · М1; доработана · М4 |
| `TensorTabularRegressionStrategy` | Регрессия · нейросетевые implementations | `tabmreg`, `ft_transformerreg`, `tab_resnetreg`, `realmlpreg` | `NeuralTabularRegressionStrategy` | `evaluation/tensor_models.py` | `strategies/models/neural.py` | Новая · М1 |
| `CuMLClassificationStrategy` | Классификация · cuML | GPU `rf`, `logit`, `svc`, `knn` | Имя сохранить | `evaluation/gpu/classification.py` | `strategies/models/cuml.py` | Была; переработана · М5 |
| `CuMLRegressionStrategy` | Регрессия · cuML | GPU `ridge`, `linear`, `rfr`, `knnreg` | Имя сохранить | `evaluation/gpu/regression.py` | `strategies/models/cuml.py` | Была; переработана · М5 |
| `CumlClusteringStrategy` | Кластеризация · cuML | GPU `kmeans` | `CuMLClusteringStrategy` | `evaluation/gpu/clustering.py` | `strategies/models/cuml.py` | Была; переработана · М5 |
| `FedotTsForecastingStrategy` | Прогнозирование временных рядов | ARIMA, AR, ETS, CGRU/CLSTM | `ForecastingStrategy` | `evaluation/time_series.py` | `strategies/models/forecasting.py` | Была |

«Целевой» означает объединение **после** согласования контрактов, не простое
переименование в уже занятое имя. До этого сохраняются отдельные реализации.
Для sklearn это требует миграции оставшихся legacy моделей. Для `NeuralTabular*`
согласовано объединение с Simple после унификации контракта implementations,
описанного в пункте 3. Отдельной стратегии/модуля `Torch*` в целевом дереве нет.

У `BoostingClassificationStrategy` / `BoostingRegressionStrategy` (XGBoost,
LightGBM, CatBoost), `TabPFN*Strategy` / `TabICL*Strategy`, `CustomModelStrategy`,
`ExtensionModelStrategy` и H2O/TPOT стратегий имена можно сохранить. Это тоже
**стратегии моделей**, а не сами модели. Они уже существовали до этих пяти коммитов
и в них не менялись; в новом дереве меняется прежде всего размещение.

### 5.3. Стратегии преобразований данных

Это тоже **стратегии**, но вызывают `transform`, а не предсказывающую модель.

| Сейчас | Вид стратегии | Примеры операций | Предлагаемое имя | Где сейчас (до переноса) | Куда планируется | В твоих пяти коммитах |
| --- | --- | --- | --- | --- | --- | --- |
| `FedotRegressionPreprocessingStrategy` | Data operations для регрессионных задач, включая табличный forecasting-путь | RFE, RANSAC, decomposition, isolation forest | `RegressionDataOperationStrategy` | `evaluation/regression.py` | `strategies/data_operations/preprocessing.py` | Была |
| `FedotClassificationPreprocessingStrategy` | Data operations для задач классификации | RFE, resampling, decomposition, isolation forest | `ClassificationDataOperationStrategy` | `evaluation/classification.py` | `strategies/data_operations/preprocessing.py` | Была |
| `FedotPreprocessingStrategy` | Общие data operations | Scaling, imputation, one-hot encoding, KernelPCA | `PreprocessingDataOperationStrategy` | `evaluation/common_preprocessing.py` | `strategies/data_operations/common_preprocessing.py` | Была |
| `TensorTransformStrategy` | Проекции / снижение размерности | `pca`, `truncated_svd` | `ProjectionDataOperationStrategy` | `evaluation/tensor_transform.py` | `strategies/data_operations/projection.py` | Была |
| `TensorOptionalPreprocessingStrategy` | Мост к optional preprocessing service | Выбор и применение шагов imputation/scaling и др. | `OptionalPreprocessingStrategy` | `evaluation/optional_preprocessing.py` | `strategies/data_operations/optional.py` | Была |
| `FedotTsTransformingStrategy` | Data operations временных рядов | `lagged`, `smoothing`, `cut`, `diff_filter` | `TimeSeriesDataOperationStrategy` | `evaluation/time_series.py` | `strategies/data_operations/time_series.py` | Была |
| `SkLearnTextVectorizeStrategy` | Векторизация текста · sklearn | `tfidf`, `cntvect` | `SklearnTextVectorizationStrategy` | `evaluation/text.py` | `strategies/data_operations/text.py` | Была |
| `FedotTextPreprocessingStrategy` | Очистка текста | `text_clean` | `TextCleaningStrategy` | `evaluation/text.py` | `strategies/data_operations/text.py` | Была |
| `GensimTextVectorizeStrategy` | Предобученные текстовые embeddings | `word2vec_pretrained` | `PretrainedEmbeddingStrategy` | `evaluation/text.py` | `strategies/data_operations/text.py` | Была |

Для этой пары выбран общий модуль `preprocessing.py`: он группирует варианты
преобразований для регрессионных и классификационных задач, как `neural.py`
группирует варианты модельных стратегий. Отдельные файлы для каждой задачи
сейчас не нужны: протокол вызова почти одинаков, различаются прежде всего mappings.

Общие преобразования — scaling, imputation, encoders и др. — размещаются в
`common_preprocessing.py` с `PreprocessingDataOperationStrategy`. Таким образом,
пара файлов различает общую подготовку и варианты для regression/classification,
не вводя префикс `Supervised` и не разделяя парные стратегии на разные модули.

`Supervised` убирается и из имён классов, и из имени модуля: не каждая входящая
операция обучается с учителем. Например, Isolation Forest сам по себе unsupervised,
а его классификационный вариант добавляет проверку сохранности классов при фильтрации.
`Regression` / `Classification` обозначают назначение группы, не способ обучения
всех её алгоритмов. Регрессионная группа также разрешена каталогом для forecasting
на табличном представлении. Эти две группы не являются полным каталогом всех
task-specific операций: текст и временные ряды сохраняют свои доменные модули.

Если семейство действительно разрастётся, его можно превратить в пакет с внутренними
модулями, но по той же политике, что и остальные семейства, а не ради симметрии
имён classification/regression. Добавлять пустые файлы или новые базовые классы
только для структуры не требуется.

`DataSourceStrategy` — отдельная **стратегия источника данных**: `data_source_table`,
`data_source_ts` и др., не модель и не обучаемый transform. Её имя можно сохранить,
размещение — `strategies/sources.py`; она тоже существовала до пяти коммитов.

### 5.4. Реализации и их базовые контракты

Здесь уже **не стратегии**: либо исполняемые модели/адаптеры/data operations, либо их базы.
Колонка примеров показывает соответствующий operation ID или наследников базы.
Это выбранные реализации, помогающие прочитать предлагаемое дерево, не полный каталог.
Отсутствие класса в таблице не означает его удаление, отсутствие переноса или
необходимость отдельного файла. При реальном переезде учитывать весь состав
исходных модулей, включая локальные базы и helpers. В частности, все RFE-варианты
из `sklearn_selectors.py` остаются связанной группой, как и существующие filters
и transformations. Перед переносом составить полную карту исходных и новых путей;
эта иллюстративная таблица не заменяет такой migration checklist.

| Сейчас | Что это | Примеры / операция | Предлагаемое имя | Где сейчас (до переноса) | Куда планируется | В твоих пяти коммитах |
| --- | --- | --- | --- | --- | --- | --- |
| `ModelImplementation` | База implementations моделей | KNN wrappers, Torch-модели и др. | Имя сохранить | `evaluation/operation_implementations/implementation_interfaces.py` | `implementations/base.py` | Была |
| `DataOperationImplementation` | База data operations; сейчас legacy-контракт | Scaling, selectors, text cleaning | Имя сохранить; перевести контракт на TensorData | `evaluation/operation_implementations/implementation_interfaces.py` | `implementations/base.py` | Была |
| `TensorDataOperationImplementation` | База TensorData data operations, не узел графа | `PCAImplementation`, `TruncatedSVDImplementation` | Целевой `DataOperationImplementation` после согласования контрактов | `evaluation/abstract_node.py` | `implementations/base.py` | Была |
| `TorchTabularModel` | База обучения native Torch-моделей | Linear/MLP classifier и regressor | Имя пока сохранить; локальная обучающая база Linear/MLP | `evaluation/operation_implementations/models/torch.py` | `implementations/models/neural.py` | Новая · М4 |
| `TorchLinearClassifier` | Реализация модели · классификация | `torch_linear` | Имя сохранить | `evaluation/operation_implementations/models/torch.py` | `implementations/models/neural.py` | Была; переписана · М4 |
| `TorchMLPClassifier` | Реализация модели · классификация | `torch_mlp` | Имя сохранить | `evaluation/operation_implementations/models/torch.py` | `implementations/models/neural.py` | Новая · М4 |
| `TorchLinearRegressor` | Реализация модели · регрессия | `torch_linear_reg` | Имя сохранить | `evaluation/operation_implementations/models/torch.py` | `implementations/models/neural.py` | Новая · М4 |
| `TorchMLPRegressor` | Реализация модели · регрессия | `torch_mlp_reg` | Имя сохранить | `evaluation/operation_implementations/models/torch.py` | `implementations/models/neural.py` | Новая · М4 |
| `TensorTabularModelImplementation` | База нейросетевых implementations | TabM / RTDL / RealMLP | Имя уточнить без обязательного объединения обучающих баз | `evaluation/operation_implementations/models/tensor_tabular.py` | `implementations/models/neural.py` | Новая · М1; доработана · М4 |
| `FedotTabMClassificationImplementation` / `FedotTabMRegressionImplementation` | Модель + интеграция TabM | `tabm`, `tabmreg` | `TabMClassifier` / `TabMRegressor` | `evaluation/operation_implementations/models/tensor_tabular.py` | `implementations/models/neural.py` | Новые · М1 |
| `FedotFTTransformer*Implementation` / `FedotResNet*Implementation` | Модели + интеграция RTDL | `ft_transformer`, `tab_resnet` и regression-варианты | Имена без `Fedot` и общего `Implementation` | `evaluation/operation_implementations/models/tensor_tabular.py` | `implementations/models/neural.py` | Новые · М1 |
| `FedotRealMLPClassificationImplementation` / `FedotRealMLPRegressionImplementation` | Модель + интеграция pytabkit | `realmlp`, `realmlpreg` | `RealMLPClassifier` / `RealMLPRegressor` | `evaluation/operation_implementations/models/tensor_tabular.py` | `implementations/models/neural.py` | Новые · М1 |
| `ExplainableBoostingClassifierAdapter` / `ExplainableBoostingRegressorAdapter` | Адаптеры моделей InterpretML, не стратегии | `ebm`, `ebmreg` | Имена сохранить | `evaluation/tensor_sklearn.py` | `implementations/models/estimator_adapters.py` | Новые · М1; перенос и доработка classifier adapter после пяти |
| `FedotKnnClassImplementation` / `FedotKnnRegImplementation` | Адаптеры sklearn KNN | `knn`, `knnreg` | `KnnClassificationAdapter` / `KnnRegressionAdapter` | `evaluation/operation_implementations/models/knn.py` | `implementations/models/estimator_adapters.py` | Были |
| `PCAImplementation` / `TruncatedSVDImplementation` | Реализации data operations | `pca`, `truncated_svd` | Имена сохранить | `evaluation/operation_implementations/data_operations/features_reducing.py` | `implementations/data_operations/features_reducing.py` | Были |
| `ComponentAnalysisImplementation` и KernelPCA/FastICA/DaskPCA-наследники | Семейная база и legacy data operations для проекций | `kernel_pca`, `fast_ica`, `dask_pca` | Сохранить группу; согласовать TensorData-контракт | `evaluation/operation_implementations/data_operations/sklearn_transformations.py` | `implementations/data_operations/features_reducing.py` | Были |
| `FeatureSelectionImplementation` и четыре RFE-наследника | База и группа data operations | `rfe_lin_reg`, `rfe_non_lin_reg`, `rfe_lin_class`, `rfe_non_lin_class` | Имена пока сохранить; не дробить группу | `evaluation/operation_implementations/data_operations/sklearn_selectors.py` | `implementations/data_operations/sklearn_selectors.py` | Были |

Суффиксы `Model` / `Adapter` / `Implementation` использовать там, где они объясняют
роль. Существующие разумные имена не нужно менять только ради переезда, в частности
`PCAImplementation` не переименовывается в `PCATransform`. `DataOperationImplementation`
сохраняет привычное имя; временную `TensorDataOperationImplementation` объединять
с ней только после согласования контрактов, не одним переименованием.
Общие интерфейсы и семейные обучающие базы не смешивать. Удаление `Fedot` или
`Implementation` из остальных имён — отдельное осмысленное решение, не обязательная
массовая замена суффиксов.

## 6. Как не смешать структуру с миграцией данных

Сейчас нельзя слить `SkLearn*` и `TensorSkLearn*`: это незаметно изменит контракт
оставшихся legacy моделей или вернёт InputData-путь у уже перенесённых моделей.
Временное сосуществование нужно показывать явно, но не закладывать в целевую структуру.

Разделять структурные изменения и согласованный шаг унификации:

- основные новые имена/пути делаются каноническими для существующих групп;
- legacy sklearn и TensorData-only sklearn размещаются вместе в `strategies/models/sklearn.py`;
- новые TensorData-only sklearn остаются отдельной реализацией стратегии;
- после изменения контрактов Linear/MLP и нормализации вероятностей обе группы
  объединяются в `NeuralTabular*`, `strategies/models/neural.py`; отдельные
  Simple/Torch стратегии не сохраняются;
- объединение sklearn-баз выполняется отдельно после завершения миграции их моделей.

Отдельный временный файл `legacy_sklearn.py` при реализации не понадобился.
После объединения контрактов меняются классы, а общий модуль остаётся на месте.

Название `NeuralTabular*` не требует вводить отдельные стратегии для каждой из
Linear / MLP / TabM / RTDL / RealMLP. Их общий вызов остаётся общим; различия алгоритмов находятся
в implementations. В пункте 4 эта группа также размещена в одном `models/neural.py`
с локальными базами; общий модуль не требует единого training loop для всех моделей.

## 7. Порядок возможного рефакторинга

1. Согласовать дерево и основные имена. Не трогать operation ID вроде `knnreg`,
   `torch_mlp`, `tabm`: они уже используются в pipeline/config/search metadata.
2. Составить полную карту переноса всех implementations, а не только строк пункта 5.4.
   Отделить базовую стратегию от sklearn; выделить соседние пакеты strategies и
   implementations. Сохранить DataOperation-терминологию и связанные группы классов.
   Для каждой ещё не перенесённой операции определить её конечную группу уже сейчас;
   отдельное место разрешать только при содержательном отличии контракта/зависимостей.
3. Разнести смешанные classification/regression/time-series модули на модели
   и data operations; уточнить `gpu` → `cuml`. Simple не переименовывать в промежуточные
   Torch-стратегии: их удаление относится к шагу объединения ниже.
4. Разместить связанные implementations моделей по групповым модулям дерева:
   нейросетевые — в `models/neural.py`, boosting — в `models/boosting.py` и т. д.
   Локальные базы переносить с их потребителями. Не дробить `tensor_tabular.py`
   по одной модели и не создавать отдельные linear.py/mlp.py без необходимости.
5. Согласовать контейнерный контракт Linear/MLP и остальных нейросетевых implementations
   по пункту 3, объединить Simple и TensorTabular в `NeuralTabular*Strategy` и
   обновить прямые и pipeline-level тесты. Миграцию оставшихся sklearn-моделей
   делать другой задачей.

При любом фактическом переезде требуется обновить:

- четыре JSON-каталога операций: они хранят строки module/class для динамического импорта;
- обычные импорты в core, industrial, tests и examples;
- Sphinx API docs и карту операций;
- тесты стратегии, metadata resolution и сохранения/загрузки pipeline.

Старые импортные пути и сохранённые пайплайны предыдущих версий в рамках этой
реструктуризации не поддерживаются. Для старых артефактов следует использовать
версию FEDOT и окружение, на которых они обучались. Миграцию сохранённого состояния
откладываем; Industrial обновит импорты при синхронизации веток.

## 8. Чего я не предлагаю

- Не заводить стратегию для каждого KNN/MLP/ResNet, если вызовы одинаковы.
- Не переносить external estimator в собственный пустой wrapper только ради каталога.
- Не прятать разный data contract за одинаковыми именами до его реального согласования.
- Не объединять все модели в универсальную стратегию с большим количеством проверок типов.
- Не заменять `Simple` на столь же неопределённые `Advanced`, `Complex` или `Deep`.
- Не вводить обязательный `*_rules.py` для каждого семейства.
- Не менять preprocessing, метрики, обучение и operation IDs одновременно с переездом.

Главный выигрыш первого этапа — по пути понятно, модель это или transform,
стратегия это или алгоритм, и какое семейство исполняется. Объединение похожих
стратегий — последующее упрощение, подтверждаемое общим контрактом, а не деревом каталогов.

**Сохранение:** предложение и фактическая карта реструктуризации включены в её коммит
явным добавлением, поскольку правило `*.md` в `.gitignore` скрывает новые Markdown-файлы.
