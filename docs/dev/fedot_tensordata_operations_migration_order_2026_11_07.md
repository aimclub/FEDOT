# Порядок переноса операций FEDOT на TensorData

7 ноября 2026. Предлагаемый порядок от более локальных адаптаций к сложным контрактам данных.
Каждый пункт — отдельная задача: переносим стратегию и реализации вместе, затем отмечаем результат.
При обнаружении зависимости порядок можно уточнить; крупный пункт допустимо разбить на подзадачи.

## Общие условия для каждого пункта

- TensorData на входе/выходе стратегии; модель заполняет `predict`, data operation — `features`.
  InputData/OutputData-путь у переносимого семейства удаляем. Предпочтительны Torch-операции и передача
  памяти без копирования; NumPy/Pandas — только необходимая граница внешней библиотеки, с описанием причины.
- Ориентир — проекции: тонкая стратегия, алгоритм в implementation, общая база только при реальном повторении.
  Сохраняем необходимые различия `predict_for_fit` / `transform_for_fit` и обычного inference.
- Не переводим числовые tensors в NumPy ради готовых sklearn/NumPy-функций, когда задачу можно компактно
  решить Torch-примитивами. Для split/масок/выравнивания строк используем общие native helpers.
  Host strings, сериализация и обязательный формат внешней библиотеки — отдельные обоснованные границы.
- Межбиблиотечный обмен выполняем через `core/backend/array_conversions.py`: as_numpy/as_cupy/as_torch,
  с явными device/dtype/copy при необходимости. Устройство выбирает вызывающий код, не global Backend
  внутри конвертера. Категории/Pool, objective и формы multioutput остаются в моделях.
  Текущие строки и список потребителей: [аудит конвертаций](fedot_boosting_conversion_audit_2026_11_10.md).
- Новые поля, функции и методы размещаем рядом с семантически связанными элементами, а не автоматически
  в конце класса/файла. Имена и описания должны явно говорить о назначении и результате.
- Сохраняем метки классов, формы результатов, idx/target и метаданные, не изменяем входной контейнер.
  Проверяем CPU, композицию узлов и режимы результатов; заявленную GPU-поддержку проверяем реальным запуском.
- GPU добавляем там, где у семейства есть подходящий механизм, с явными настройками устройства и переносов.
  Не создаём отдельные CPU/GPU-классы без причины. Общую политику `auto` и объединение каталогов пока откладываем.
- В рамках пункта обновляем соответствующие тесты, импорты/регистрацию и карту форматов; фиксируем непроверенные режимы.
  Настройки метрик, preprocessing и алгоритма не меняем незаметно в процессе переноса.
- Перед изменением нескольких файлов перечисляем их; в результате каждого пункта приводим пользователю
  список изменённых файлов, принятые решения и фактический путь данных: Torch-native, zero-copy или host adapter.
  Краткую запись о формате оставляем непосредственно под выполненным пунктом этого документа.
- Сохраняем всю существующую функциональность обучения. По возможности до правок снимаем небольшой
  детерминированный InputData-baseline и сравниваем с TensorData после переноса; существенные различия разбираем.
  Обнаруженные старые дефекты, меняющие результаты обучения, обсуждаем отдельно от смены формата.
- Проверяем не только отдельный fit, но и взаимодействия параметров модели с optional preprocessing,
  категориями, пропусками, индексами и fit/predict. Смотрим прежний путь: полезную семантику сохраняем,
  старые ошибки не воспроизводим автоматически; исправления, влияющие на обучение, явно согласовываем.
- Для внешних библиотек проверяем синонимы параметров: framework defaults и пользовательские настройки
  не должны одновременно задавать один параметр разными именами. Особенно проверяем отключение eval set.
  Для multioutput проверяем значения и оси каждого target, а не только возможность fit;
  состояние одного обучения не должно записываться в общий класс реализации.
- При проверке early stopping различаем число выполненных раундов и число деревьев в итоговой модели /
  predict. Большая patience не отключает выбор best iteration. Для маленьких данных отдельно учитываем
  уменьшение train из-за внутреннего eval split; политику split/defaults не меняем незаметно при переносе.
- Вопросы выбора настроек, требующие экспериментов на данных, пополняем отдельно в
  [списке экспериментов](fedot_model_experiment_questions_2026_11_10.md). Не смешиваем такие изменения
  качества с исправлением явных ошибок и контейнерной миграцией.
- Не усложняем код ради формального переноса. Существенный рост, неясный контракт или нетипичный workaround
  подсвечиваем пользователю до продолжения зависимой части. Общую повторяющуюся логику можно вынести,
  при необходимости поправив уже перенесённые семейства; такие изменения также перечисляем.
- Если готового GPU-пути нет или для него нужна существенная отдельная реализация, перенос на TensorData
  выполняем для CPU; по согласованию новые GPU-классы пока откладываем. По мере встречи моделей пополняем
  [список отложенной GPU-поддержки](fedot_cpu_gpu_operations_checklist_2026_11_07.md#отложенная-gpu-поддержка-при-миграции-моделей).
- Проверки ограничиваем короткими CPU/GPU-запусками и адресными проверками сохранения поведения.
  Полные поисковые/нагрузочные прогоны сейчас не нужны. Если GPU скрыт sandbox, запрашиваем запуск вне sandbox;
  отсутствие GPU в sandbox не считаем отсутствием видеокарты у пользователя.

## Чек-лист

Уточнение приоритета: перенос TS-моделей пока отложен по просьбе пользователя.
После табличных моделей продолжаем табличные data operations; forecasting-модели и TS-нейросети
остаются в списке для возвращения к ним позднее.

- [x] **1. Бустинги: CatBoost, XGBoost, LightGBM, классификация и регрессия.**
  `strategies/models/boosting.py` + `implementations/models/boosting.py`.
  Перенести категориальные признаки, подготовку target, eval set/early stopping и multioutput.
  Добавить явный CPU/GPU-режим через параметры библиотек, учитывая их различия; одной реализации обычно достаточно.
  **Начато, 7 ноября:** сохранён CPU InputData-baseline для трёх библиотек: binary/multiclass/regression,
  validation/no validation и multioutput (`/tmp/fedot_boosting_before.{json,npz}`;
  воспроизводящий скрипт `/tmp/fedot_boosting_baseline_2026_11_07.py`). Исходники моделей пока не изменены.
  Старый `get_not_encoded_data()` возвращает исходные столбцы, а TensorData уже может содержать OHE.
  **Реализована подготовительная часть 9 ноября:** добавлены `categorical_features` (значения по строкам)
  и `categorical_source_idx` (исходные ID столбцов), плюс `get_not_encoded_data()` без inverse encoding.
  Существующие `categorical_idx`/`numerical_idx` относятся к текущим столбцам, `idx_mapping` — к их происхождению.
  По согласованию imputation категорий обновляет исходное и кодированное представления вместе;
  без imputation пропуск остаётся пропуском. Создание данных обязательного заполнения не выполняет.
  Числовые признаки остаются Torch; исходные строки — NumPy sidecar. `CategoricalData` — небольшой тип
  в `contracts.py`, без pandas-конверсии. Формат внешней библиотеки определит адаптер бустинга на следующем этапе.
  Контракт, проверки и полный список файлов: [категории TensorData](fedot_tensordata_categorical_features_2026_11_09.md).
  **Повторная проверка 9 ноября, до миграции:**
  - Прежние `/tmp/fedot_boosting_before.*` больше недоступны; baseline предстоит снять заново.
  - CatBoost со стандартными параметрами и `use_eval_set=False` падает: обёртка добавляет
    `early_stopping_rounds=False` при уже заданном `od_wait`; CatBoost отклоняет два синонима.
  - Multioutput binary, 120 строк / 2 target: CatBoost и XGBoost возвращают из модели `(120, 2)`,
    но стратегия для `probs` оставляет `(120,)`, теряя один target. LightGBM `full_probs` даёт
    `(2, 120, 2)` вместо оси samples первой. `is_multi_target` также записывается в класс реализации.
    Исправление контрактов требует согласования, потому что результаты отличаются от старых дефектных.
  - XGBoost принудительно отключает `enable_categorical` для `gbtree`. В установленном XGBoost 3.0.5
    native categories с `hist` поддерживаются. При снятии запрета нужно явно согласовать эффективный default:
    в JSON сейчас True, фактически в старом коде False.
  - **Уточнение о LightGBM:** отсутствие append callback не отключает early stopping. Библиотека сама создаёт
    callback из параметра `early_stopping_rounds`. Проверено: при 40 деревьях / patience=2 остановилась
    с best_iteration=4. Не добавлять второй callback и не объявлять это неработающим early stopping.
  Исходники стратегий/реализаций бустингов на этой проверке не менялись.
  **Разбор истории и early stopping, 10 ноября:** `6621f8d3` (#1410) и локальный `d0a4b45f`
  не меняли бустинги: изменения default params относились к RANSAC. CatBoost-подстановка
  `early_stopping_rounds=False` присутствовала уже на том срезе; blame на `1efbb4f3` объясняется
  форматированием, а не появлением этой логики. Нельзя считать её регрессией #1435 без проверки diff.
  Связанный исторический `6dd93b25` (#1358) добавлял корректировки при фактическом отсутствии eval set
  и заменял False на None для LightGBM; CatBoost-конфликт одновременных aliases там не устранён.
  На LightGBM 4.6.0 воспроизведено с одинаковым train/eval и 24 раундами: patience=None даёт 24 раунда
  в итоговом booster; patience=30 и 1e9 обе выполняют 24 eval-итерации, выбирают best_iteration=1
  и сохраняют booster с одной итерацией. Это объясняет возможное отличие качества даже без раннего
  завершения цикла; конкретные ligands-эксперименты пользователя заново не воспроизводились.
  **10 ноября: согласованы и исправлены три подготовительные проблемы, без миграции бустингов:**
  - CatBoost defaults используют `early_stopping_rounds=100` вместо пары `od_type=Iter` / `od_wait=100`.
    Эффективный режим Iter / ожидание 100 и `use_best_model=True` сохранены. При реальном отсутствии eval set
    параметры детектора удаляются, выбор best model отключается в одной существующей no-eval ветке.
  - При объединении defaults явный `od_wait` вытесняет default `early_stopping_rounds`; оба явно переданных
    синонима отклоняются до библиотечного fit. Нативный IncToDec по-прежнему доступен явно.
  - Multioutput хранится в экземпляре, а loss настраивается в копии параметров. Вероятности согласованы:
    binary `probs/default` — `(samples, targets)`, `full_probs` — `(samples, targets, classes)`.
    Для одинакового числа классов поддержано и multiclass; ragged-контракт отложен и записан в эксперименты.
  - XGBoost больше не сбрасывает явный `enable_categorical=True`. Default записан как False, что сохраняет
    прежнее эффективное поведение стандартного gbtree. Его обученные category dtypes сохраняются для eval /
    predict: в XGBoost 3.0.5 повторное построение pandas-кодов иначе меняло прогноз одной категории
    в зависимости от состава batch. Unknown категории используют missing-value semantics библиотеки.
  - Сняты числовые single-target baseline для всех шести моделей; labels/probs/full_probs и regression
    predictions до и после исправлений совпали точно. Исходные результаты — временные `/tmp/*three_fixes_before.*`.
    37 адресных CPU-тестов параметров и boosting contracts прошли; GPU/full search для этих исправлений не запускались.
  На этапе этих трёх исправлений реализации ещё оставались на InputData-пути.
  Удаление legacy-узлов encoding потребует обновления каталогов, примеров и встроенного Industrial;
  исходники этих узлов пока не удалены. Также отдельно отмечен неиспользуемый callback LightGBM.
  На первоначальном этапе по библиотечным границам: XGBoost CUDA — DLPack/CuPy, CatBoost/LightGBM — host API;
  полного Torch-native обучения этих внешних алгоритмов без смены алгоритма нет.
  **Перенос завершён 10 ноября:** все шесть моделей и обе стратегии — TensorData-only.
  Общая база `BoostingImplementation` объединяет split, подготовку target/categories и Torch-результат.
  Первоначально стратификация совпадала с InputData; позже заменена Torch helper с той же политикой,
  но другим RNG. Последовательный default общего tensor split не изменён.
  Обученный LabelEncoder для исходных категорий переиспользуется между train/eval/predict, pandas больше нет.
  CatBoost использует FeaturesData/Pool, пропуск без imputation передаётся строковым маркером `nan`;
  XGBoost/LightGBM получают числовые коды категорий с NaN для пропусков/неизвестных значений.
  **Формат:** подготовка и упаковка Torch; XGBoost CPU/CUDA-входы теперь передаются как Torch напрямую;
  LightGBM и CatBoost — необходимые NumPy/host adapters. CUDA output XGBoost теперь возвращается через
  CuPy → Torch DLPack; CPU/gblinear сохраняют sklearn-путь. Сам алгоритм обучения остаётся библиотечным.
  **Проверки:** 21 короткий fit / 45 массивов предсказаний точно совпали с InputData-baseline;
  80 адресных CPU-тестов; 9 GPU fits на RTX 4080 SUPER — classification/regression/native categories
  для каждого семейства, с eval set. Проверена связка optional preprocessing → boosting.
  Политика defaults/eval split/patience и CPU/GPU-каталоги не изменены; полные AutoML-прогоны не выполнялись.
  Файлы, ограничения и параметры GPU: [результат переноса](fedot_tensordata_boosting_2026_11_10.md).
  **Уточнение 10 ноября:** прямой Torch работает уже в XGBoost 3.0.5; NumPy/CuPy-конвертер входов удалён.
  Дополнительная проверка: 12 CPU-тестов, 3 GPU fits, точное совпадение 15 массивов с InputData-baseline.
  Выходной NumPy пока обусловлен sklearn API; Booster.inplace_predict может вернуть CuPy на CUDA,
  что проверено отдельно. Эту оптимизацию нельзя путать с уже устранённой конверсией входов.
  **Дальнейшая доработка 10 ноября выполнена:** стратификация и Torch row alignment — без host roundtrip;
  GPU output XGBoost включён в основную реализацию. 106 CPU / 16 GPU адресных тестов, новые замеры и аудит:
  [native rows](fedot_tensor_native_rows_2026_11_10.md).

- [x] **2. Оставшиеся sklearn-модели и clustering.**
  `strategies/models/sklearn.py`: линейные модели, деревья/леса, Naive Bayes, MLP, KMeans и остальные текущие ID.
  Использовать уже существующий TensorData-адаптер; после полного объединения убрать временное разделение
  `TensorSkLearn*`/обычных стратегий. Сохранить новые ExtraTrees/HistGB/EBM, не переносить их повторно.
  Проверить аналогичную запись `self.operation_impl.is_multi_target` в `SklearnModelStrategy.fit`:
  состояние отдельного fit должно принадлежать обученному экземпляру, как исправлено у бустингов.


  **Выполнено 10 ноября:** единые SklearnModelStrategy / Classification / Regression / Clustering,
  23 ID (включая шесть ранее перенесённых), без TensorSkLearn* и InputData/OutputData-пути.
  Сохранены параметры, random state, singleton target, native multioutput и wrappers по тегу non_multi,
  а также default KMeans n_clusters=2. is_multi_target теперь принадлежит обученному экземпляру.
  Исправлена обработка вероятностей multioutput classification: samples-first, все target сохранены;
  native MLP multilabel нормализуется в тот же контракт. Ragged class axis (разное число классов по target)
  не добавлен: TensorData.predict — плотный tensor, такое представление нужно согласовывать отдельно.
  Дополнительные HistGB/EBM classification по-прежнему не получили новый multioutput-контракт;
  миграция не добавляет им новую функциональность и не меняет метаданные non_multi.
  **Формат:** внешние CPU sklearn/InterpretML estimators получают NumPy через as_numpy;
  предсказания становятся Torch через as_torch на device входа. Преобразование — граница библиотеки,
  а не переписывание sklearn-алгоритмов на Torch. Probability stacking выполняется в Torch.
  По согласованию сохраняем единый NumPy-путь и для Ridge: отдельный экспериментальный Array API
  путь сейчас не вводим, чтобы не усложнять общий контракт семейства.
  CUDA-вход означает CPU-fit sklearn и возврат результата на CUDA; cuML и выбор CPU/GPU не менялись.
  **Проверка поведения:** до изменения сняты 54 prediction baseline для всех 23 ID, включая multioutput
  regression; после переноса все совпали точно. Скрипт /tmp/fedot_sklearn_parity.py,
  baseline /tmp/fedot_sklearn_baseline.npz (временные файлы, не долгосрочное хранилище).
  Адресные тесты: tests/core/operations/test_sklearn_tensordata.py и sklearn/EBM-часть
  tests/core/operations/test_experimental_tabular_models.py; включают pipeline, pickle текущего estimator,
  разные output_mode, независимость моделей и CUDA-boundary smoke.
  Результаты: 47 CPU-тестов + 3 дополнительных multiclass; 5 CUDA smoke-проверок прошли.
  **Файлы для ревью:**
  - fedot/core/operations/strategies/models/sklearn.py — единый TensorData-контракт и mappings по задачам;
  - fedot/core/repository/data/model_repository.json — единые metadata, прежние ID/presets/параметры;
  - fedot/industrial/core/operation/interfaces/industrial_preprocessing_strategy.py — legacy-helper локально;
  - два упомянутых файла тестов;
  - эта записка, fedot_operations_data_formats_2026_10_09.md,
    fedot_operations_structure_proposal_2026_10_09.md, fedot_industrial_tensordata_sync_2026_11_05.md.

- [x] **3. FEDOT-адаптеры табличных моделей.**
  `strategies/models/adapters.py` + `implementations/models/estimator_adapters.py`:
  KNN classification/regression, SVC, LDA, QDA. CNN из этой стратегии переносится отдельно в п. 13.


  **Выполнено 10 ноября:** AdapterModelStrategy объединяет fit, classification/regression получают
  TensorData и возвращают TensorData через существующий EvaluationStrategy._replace_predict_in_tensor_data.
  Общий SklearnModelAdapter содержит необходимую CPU-estimator границу; его реализации возвращают Torch.
  Сохранены округление/коррекция k, OneVsRestClassifier у SVC, коррекция shrinkage и fallback lsqr у LDA,
  формы KNN regression target (1D / один столбец / multioutput), nan_to_num у discriminant analysis.
  **Формат:** as_numpy только перед sklearn.fit/predict/predict_proba; as_torch возвращает результат
  на device входных данных. Unique классов, nan_to_num и нормализация вероятностей выполняются в Torch.
  KNN multioutput probabilities складываются samples-first; SVC multilabel восстанавливает class axis,
  чтобы стратегия больше не теряла target. Общий ragged-контракт не вводился.
  CNN отделён в KerasClassificationStrategy, strategies/models/keras.py, с прежним InputData-контрактом.
  CPU/GPU registry merge не делался; существующие cuML KNN/SVC сохранены. LDA/QDA и CNN записаны
  в список отложенной GPU-поддержки, без новых ограниченных GPU-ветвей.
  **Проверка:** 57 сравнений предсказаний и параметров с InputData-baseline совпали точно;
  binary/multiclass, singleton/multioutput regression, k correction и LDA fallback.
  Скрипт /tmp/fedot_adapter_parity.py и /tmp/fedot_adapter_baseline.npz — временные артефакты.
  Адресные тесты: 36 CPU-тестов (включая canonical strategies / pipeline serialization), 5 CUDA boundary
  smoke-проверок; все прошли. Отдельно проверены output_mode, inference без target, metadata,
  pickle, multioutput classification, fit-stage hook и прежний CNN route через stub без обучения CNN.
  Старый integration module test_model.py не собирается из-за существующего module-level skip
  в импортируемом test_time_series_operations.py; LDA fallback проверен отдельным новым тестом.
  **Файлы для ревью:**
  - fedot/core/operations/strategies/models/adapters.py — TensorData-стратегии;
  - fedot/core/operations/implementations/models/estimator_adapters.py — общая граница sklearn и пять моделей;
  - fedot/core/operations/strategies/models/keras.py — отделённая legacy CNN-стратегия;
  - fedot/core/repository/data/model_repository.json — CNN metadata отдельно, adapters помечены tensor-only;
  - tests/core/operations/test_estimator_adapters_tensordata.py — новые проверки;
  - test/integration/models/test_model.py — прямой LDA regression case получает TensorData;
  - эта записка, карта форматов, CPU/GPU-чек-лист и заметка синхронизации Industrial.

- [x] **4. Foundation-модели: TabPFN и TabICL.**
  Стратегии и реализации `models/foundation.py`: TensorData, категории, классы и CPU/GPU-режимы.
  Разобрать дублирующие `max_samples`/`max_features` по CPU/GPU-записке; текущие GPU-ID пока направить
  на те же перенесённые реализации, без преждевременного объединения каталогов.


  **Выполнено 10 ноября:** TensorData-only для четырёх моделей и четырёх существующих GPU-aliases;
  реализации возвращают Torch, стратегии прикрепляют его к исходной TensorData без OutputData/bridges.
  Unique классов и разворачивание singleton target выполняются в Torch. Входной контейнер не изменяется.
  **Формат внешних библиотек:** установлены TabPFN 2.2.1 и TabICL 2.1.1. Обычный public fit/predict
  использует CPU NumPy/sklearn preprocessing и возвращает NumPy, даже если transformer вычисляет на CUDA.
  as_numpy/as_torch сосредоточены на этой границе; CPU-массивы разделяют числовое хранилище, где возможно.
  Не включали differentiable_input TabPFN: он меняет preprocessing/ensemble policy, а не просто формат.
  Внутренние _raw_predict / forward не подменяют public predict: иначе пришлось бы копировать native
  preprocessing, rounding/normalization probabilities и regression distribution postprocessing.
  Установленные библиотеки и веса не обновлялись, новые версии/модели не подставлялись.
  **Категории:** TabPFN enable_categorical восстанавливает числовой Torch-блок и исходный host sidecar,
  собирая mixed object NumPy только перед библиотекой. Индексы категорий вычисляются после восстановления:
  старый код присваивал их до перестановки столбцов, что могло помечать numerical column как category.
  Пропуски/unknown передаются native preprocessing; optional imputation TensorData не отменяется.
  TabICL по-прежнему получает подготовленное encoded numeric представление; новый raw-category режим не добавлен.
  **Лимиты и device:** существующие max_samples/max_features сохранены, ничего не сэмплируется автоматически.
  Удаление искусственных caps и объединение CPU/GPU defaults остаются решением общей CPU/GPU-задачи;
  библиотечный ignore_pretraining_limits по-прежнему не отключает отдельный FEDOT check_data_size.
  **Проверки:** 14 CPU-baseline predictions совпали точно (binary/multiclass/regression обеих библиотек);
  четыре реальных CUDA-запуска через GPU-ID совпали с native API и вернули CUDA Torch.
  Ещё два реальных CPU-запуска TabPFN classification/regression проверили raw categories, missing/unseen
  и исправленные индексы. 25 адресных CPU-тестов, включая проверки canonical catalogue, прошли.
  Все реальные проверки использовали уже скачанные веса в offline-режиме, n_estimators=1 и маленькие данные.
  Временные артефакты: /tmp/fedot_foundation_parity.py, /tmp/fedot_foundation_baseline.npz,
  /tmp/fedot_foundation_cuda_smoke.py, /tmp/fedot_foundation_category_smoke.py.
  **Файлы для ревью:** strategies/models/foundation.py, implementations/models/foundation.py,
  model_repository.json, gpu_models_repository.json, tests/core/operations/test_foundation_tensordata.py,
  test/integration/models/test_strategy.py (подготовка TensorData только в test helper),
  эта записка, карта форматов и CPU/GPU-чек-лист. Существующий EvaluationStrategy helper переиспользован.

- [x] **5. Обычные табличные преобразования и data sources.**
  `common_preprocessing.py`, `sklearn_transformations.py`, `categorical_encoders.py`, `strategies/sources.py`:
  scaling, normalization, imputation, polynomial features, OHE/label encoding и источники данных.
  Учесть fitted-схему категорий, имена/индексы столбцов и взаимодействие с уже готовым optional preprocessing.
  **Частично выполнено 9 ноября:** исходные категории и согласованная optional imputation уже реализованы;
  legacy-узлы OHE/label пока сохранены до переноса оставшихся InputData-зависимостей.
  **Подход к реализации:** для простых числовых преобразований предпочесть native Torch CPU/CUDA,
  а не адаптеры Torch → sklearn → Torch. В preprocessing/methods уже есть Torch scaling/normalization
  и imputation: сначала проверить переиспользование общей математики для optional preprocessing и нод.
  Сохранить FEDOT-правила выбора столбцов, обработки binary/OHE, пропусков и metadata.
  **Согласовано 11 ноября:** сохранять исходный порядок; столбцы с <=2 уникальными значениями не
  масштабировать, остальные (включая многозначные label-коды) масштабировать. Возврат raw categories
  у модели отменяет изменение только их encoded представления. При imputation обновлять обе формы.
  PCA раньше находился вместе со scaling/imputation в FedotPreprocessingStrategy; TensorTransformStrategy
  выделен в 27b3f2aa (#1446) при частичном переходе на TensorData. Нынешнее разделение
  PreprocessingDataOperationStrategy / ProjectionDataOperationStrategy — следствие этой миграции.
  Цель — единая TensorData PreprocessingDataOperationStrategy для обычных fit/transform-операций,
  включая PCA/SVD; отдельную стратегию проекций и дополнительную базу ради двух одинаковых dispatchers
  не сохранять. До переноса оставшихся реализаций учитывать временную несовместимость контрактов.
  **Уточнение legacy-поведения, 11 ноября:** воспроизведена реальная рассинхронизация InputData
  scaler → DataMerger → get_not_encoded_data: после [numeric, OHE] → [OHE, scaled numeric]
  numerical_idx/encoded_idx остаются прежними, восстановление может потерять numerical column.
  Отдельная старая imputation-нода заполняет encoded features, но сохраняет пропуск в categorical_features.
  Глобальный старый preprocessing избегал второго дефекта, когда выполнял imputation до encoding:
  data_has_categorical_features заново сохранял категории из уже заполненных features.
  В TensorData сохранять согласованное заполнение raw categories + их encoded block; оно уже реализовано
  в CategoricalImputation. Воспроизводящий скрипт: /tmp/fedot_legacy_preprocessing_explanation.py.


  **Выполнено 11 ноября:** единый PreprocessingDataOperationStrategy; projection.py и отдельный
  ProjectionDataOperationStrategy удалены, PCA/SVD используют общий dispatcher. DataSourceStrategy
  возвращает TensorData напрямую. Реализации scaling/normalization/poly_features собраны в tabular.py;
  прежний sklearn_transformations.py удалён, импорты обновлены.
  **Формат:** scaling/normalization/poly_features полностью Torch на CPU/CUDA. Матрица значений в NumPy
  не переводится; scaler-математика общая с optional handlers. Polynomial terms — Python index metadata
  + Torch products; прежний выбор до 10 исходных колонок и interaction_only/degree сохранён.
  Non-binary selector повторяет np.unique (NaN считается одним значением). Прежний nan_to_num(...,0)
  после transform сохранён; это отдельная политика, не новая автоматическая imputation raw categories.
  Порядок колонок исправлен по согласованию, source mapping/имена/категории теперь соответствуют выходу.
  Преобразования не меняют вход, сбрасывают predict и fingerprint при изменении features.
  **Удаление дублей:** old OHE/label pipeline nodes и categorical_encoders.py удалены из core и
  встроенного Industrial-каталога/исключений. Obligatory TensorData encoders остаются в preprocessing/methods.
  simple_imputation убран из табличного поиска: там используется optional_preprocessing. Старый InputData
  ImputationImplementation сохранён для TS/Industrial и зарегистрирован через существующую TS-стратегию;
  его legacy sidecar-дефект отдельно сейчас не переписывался. TS-модели не переносились.
  **Optional после промежуточных нод:** accepted_node_types=any вместо primary, чтобы не потерять
  imputation после, например, PCA. Сам imputer/его defaults не заменены. OptionalService.predict вычисляет
  input hash до преобразования, если fingerprint законно сброшен; работает на собственной копии features
  и metadata, очищает predict. Это предотвращает Hasher.hash(None), изменение соседней ветки и stale predictions.
  **Проверка:** 26 массивов совпали с InputData baseline с точностью 1e-12 после согласованной обратной
  перестановки старых результатов; float32/float64, polynomial degree/interaction и category recovery проверены.
  Прошли 56 PCA/SVD/catalogue-тестов, 80 проверок tabular/optional/categories и дополнительные dtype cases;
  9 CUDA-тестов проверили GPU и CPU↔GPU inference. Скрипты /tmp/fedot_preprocessing_baseline.py,
  /tmp/fedot_compare_native_preprocessing.py и baseline /tmp/fedot_preprocessing_baseline.npz — временные.
  **Файлы для ревью:** strategies/data_operations/common_preprocessing.py, strategies/sources.py,
  implementations/data_operations/tabular.py, preprocessing/methods/scaling_normalization.py,
  preprocessing/service/optional_service.py, data_operation_repository.json;
  tests/core/operations/test_tabular_preprocessing_tensordata.py, обновлённые projection/fit-stage tests.
  Дополнительно: legacy TS import/registration, Industrial imports/catalogue, API catalogue docs/preset cleanup,
  примеры и устаревшие encoder-node tests. Полный перечень виден в git diff; не включать benchmark result dirs.

- [x] **6. Оставшиеся проекции — TensorData-контракт.**
  KernelPCA, FastICA, DaskPCA в `implementations/data_operations/features_reducing.py` и их диспетчеризация.
  Выполнено вместе с объединением dispatcher 11 ноября: ComponentAnalysisImplementation теперь
  TensorData-only, as_numpy/as_torch используются только у границы sklearn/Dask. Параметры, fallback
  в обычный PCA и алгоритмы сохранены. Реальный CPU/GPU boundary parity подтверждён для всех трёх.
  Новые native GPU-алгоритмы для них не реализованы; это зафиксировано в CPU/GPU-чек-листе.

- [x] **7. Отбор признаков.**
  Выполнено 11 ноября: rfe_lin_class/rfe_non_lin_class/rfe_lin_reg/rfe_non_lin_reg принимают и возвращают
  только TensorData. Подключены к общей PreprocessingDataOperationStrategy; из legacy task-specific
  dispatchers удалены. Ограничения задач, пресетов и non-default сохранены отдельными metadata-группами.
  **Формат:** только sklearn RFE.fit получает features/target через as_numpy; support_ возвращается через
  as_torch. Nonzero, маска и выбор колонок — Torch. Transform не вызывает CPU/NumPy-конверсии матрицы;
  переносит на устройство inference лишь небольшой список выбранных колонок. Числа не проходят через pandas.
  **Общее:** TensorData.subset_features выбирает prepared columns и обновляет numerical/categorical indices,
  имена и idx_mapping, очищает predict/fingerprint, сохраняет idx/target и не меняет вход. Host string sidecar
  обрезается по surviving source IDs. Внутренний порядок source IDs сохраняется; Python lists — метаданные.
  Если от OHE осталась часть колонок, enable_categorical восстанавливает целый исходный категориальный признак;
  полностью удалённые категории не восстанавливаются. Row layout и fitted preparation state не меняются.
  **Сохранённая логика:** одиночный признак проходит без RFE; LogisticRegression при нескольких targets
  использует первый, как прежний fallback, деревья и LinearRegression сохраняют multioutput. Широкий
  except ValueError удалён: посторонняя ошибка не вызывает второй fit и не маскируется ошибкой индексирования.
  Частичные параметры больше не подставляют step=None: отсутствующие keys используют прежние sklearn defaults.
  Повторный fit после single-column input больше не оставляет старый passthrough flag.
  **Взаимодействия:** optional numerical imputation после selection корректно разрешает source names/IDs;
  mode imputation частичного OHE обновляет raw category и оставшиеся bits согласованно. Существующее ограничение
  unseen constant + неполный OHE-блок сохранено: явная ошибка вместо восстановления удалённых dummy columns.
  **Проверка:** 24 InputData baseline случая (4 операции × vector/column/multi target × defaults/custom params)
  дали точно те же masks и values. Прошла 81 адресная CPU-проверка: fit-stage, pipeline, serialization,
  параметры, category recovery, optional preprocessing и неизменность входа. 13 CUDA-проверок прошли:
  CPU sklearn fit с CUDA input и Torch transform CPU↔GPU; это не GPU-обучение RFE. GPU RFE отложен.
  Временные файлы: /tmp/fedot_rfe_before.py, /tmp/fedot_rfe_baseline.py, /tmp/fedot_rfe_baseline.npz,
  /tmp/fedot_compare_rfe.py; результаты /tmp/fedot_rfe_after.log и /tmp/fedot_rfe_gpu_tests.log.
  **Файлы для ревью:**
  - fedot/core/data/tensor_data/tensor_data.py;
  - fedot/core/operations/implementations/data_operations/sklearn_selectors.py;
  - fedot/core/operations/strategies/data_operations/common_preprocessing.py и preprocessing.py;
  - fedot/core/repository/data/data_operation_repository.json;
  - tests/core/data/test_tensor_feature_selection.py;
  - tests/core/operations/test_feature_selection_tensordata.py.
  Документация: этот checklist, карта форматов, список GPU-долгов, записка Industrial и вопросы экспериментов.

- [x] **8. Фильтры и resampling.**
  Выполнено 11 ноября: ransac_lin_reg/ransac_non_lin_reg, isolation_forest_reg/class и resample —
  TensorData-only, общий PreprocessingDataOperationStrategy. Legacy task-specific стратегии теперь
  содержат только decompose. Metadata для уже перенесённых supervised операций обобщены до
  classification_data_operations/regression_data_operations; задачи, пресеты и прочие теги сохранены.
  sklearn tag сохранён для библиотечных RFE/filters, у нативного resample больше не применяется.
  **Общий выбор строк:** TensorData.subset_rows использует существующий take_rows и согласованно
  выбирает/повторяет features, target, predict, idx и categorical_features; fingerprint сбрасывается.
  Только позиции для host labels/категорий переходят на CPU; числовые tensors остаются на устройстве.
  Операции очищают predict, не меняют вход и сохраняют column metadata/source mapping.
  **RANSAC:** sklearn fit получает CPU arrays один раз до retry loop; расчёт шага residual_threshold,
  проверка доли inliers и применение mask — Torch. Прежние 10 попыток, ratio>=0.4 и fallback на все строки
  сохранены, включая обработку ValueError/LinAlgError. Внутренний StandardScaler(with_mean=False)
  линейного estimator остаётся внутри sklearn: он обучается заново на каждом consensus subset.
  residual_threshold=None теперь явно вычисляет прежний sklearn MAD в Torch, чтобы при необходимости
  увеличить порог, не получить None + step. Framework defaults (threshold=0.5) не менялись.
  **Isolation Forest:** sklearn fit/predict остаются CPU boundaries; target в fit не передаётся (sklearn
  его игнорирует). Сравнение prediction==1, защита классов и выбор строк — Torch. Сохранение всех классов
  и минимум 20% каждого проверяются до создания subset, поэтому пустая маска даёт прежний passthrough.
  **Resample:** native Torch CPU/CUDA: unique/counts, row positions, randint с replacement либо randperm
  без replacement. Minority-first порядок, round, automatic replacement при n_samples>=pool size,
  пропуск balanced/nonbinary задач и поправки ratio сохранены. Старое ratio=0/None/out-of-range означает
  full balance вопреки прежнему docstring; это отмечено для отдельного решения без смены поведения сейчас.
  После реального resampling idx по-прежнему сбрасывается в 0..N-1; дубликаты имеют разные output labels.
  Raw categories повторяются теми же positions. Target сохраняет исходные dtype/column shape, вместо
  старого смешивания с features в NumPy и неявного преобразования labels во float/flatten.
  Torch RNG не воспроизводит последовательность NumPy. Алгоритмы сравнены при одинаковых random draws;
  в pipeline используется fitting seed через локальный Torch generator, без изменения global Torch RNG.
  Inference всех пяти операций сохраняет исходное число строк и не вызывает sklearn.
  **Проверка:** 16 InputData filter cases точно совпали по rows/target/idx, threshold — до 1e-12.
  40 resample configurations точно совпали по значениям/idx/коррекции параметров при shared draws.
  106 адресных CPU-проверок, 25 перенесённых legacy-regressions и 6 реальных CUDA-проверок прошли;
  проверены категории, fit/inference, mixed filtered branches, repeatability и запрет cpu/numpy/tolist
  для числового CUDA resample. Это не GPU fit sklearn filters; сложные GPU реализации отложены.
  Baselines/scripts: /tmp/fedot_filters_resample_baseline.py, /tmp/fedot_filters_baseline.npz,
  /tmp/fedot_resample_baseline.npz, /tmp/fedot_compare_filters.py, /tmp/fedot_compare_resample.py.
  **Файлы для ревью:**
  - fedot/core/operations/implementations/data_operations/sklearn_filters.py и resampling.py;
  - fedot/core/data/tensor_data/tensor_data.py (subset_rows);
  - fedot/core/operations/strategies/data_operations/common_preprocessing.py и preprocessing.py;
  - fedot/core/repository/data/data_operation_repository.json;
  - tests/core/operations/test_filters_resampling_tensordata.py;
  - tests/core/data/test_tensor_row_operations.py;
  - test/unit/data_operations/test_sklearn_filters.py, test_data_operation_params.py,
    test_data_operations_implementations.py (существующие regression cases переведены на TensorData).
  Документы: этот checklist, карта форматов, GPU-долги, записка Industrial и вопросы экспериментов.

- [x] **9. Decompose для классификации и регрессии.**
  `implementations/data_operations/decompose.py` + `strategies/data_operations/preprocessing.py`.
  Разобрать несколько родителей, признаки/предсказания и изменение target без legacy supplementary_data.
  Перед изменением merger/выравнивания свериться с запиской Industrial.
  **Реализация 11 ноября:** выбран упрощённый graph-context подход, без новых полей TensorData.
  tensor_input_rules.py вычисляет main/auxiliary и глубину по предыдущим nodes; tensor_input.py
  подготавливает вход decompose до concat. Merger принимает выбранный target source и выравнивает
  данные, арифметики decompose в нём нет. Операции подключены к общему PreprocessingDataOperationStrategy,
  последние legacy task-specific dispatchers и strategies/data_operations/preprocessing.py удалены.
  **Формат:** residuals, class mapping и alignment — Torch CPU/CUDA; sklearn/числовой NumPy для операции нет.
  Features + их исходные category metadata берутся от выбранной ветви; baseline передаётся в predict,
  output predict очищается. Вход не мутируется. Fitted class vocabulary хранится у decomposer, не в data.
  Обычный merge сохраняет main priority; все auxiliary — прежний shallowest fallback. Fork после decompose
  обучает внутренний ensemble на residual target, внешний main merge — на исходном y. Inference без y
  сохраняет правильный task; provided y не очищается автоматически (legacy convention сохранён).
  Fingerprint учитывает baseline values и сбрасывается после изменения входа/выхода. Generic tabular
  column metadata обновляется и без raw sidecar; multidimensional TS/image layout пока не переделывался.
  **Согласованный binary fix:** legacy argwhere bug всегда делал первую метку positive. Пользователь
  подтвердил некорректность поведения; исправление сохранено. При p=0.2 отрицательный объект даёт -0.2,
  положительный +0.8. Прямой test подтверждает, что следующая linear-модель учится на этих signed residuals
  в regression task, а финальная модель получает исходный target. Из 16 baseline cases 12 совпали точно;
  в 4 binary cases единственное отличие — исправленная первая negative row (разница ровно 1).
  Финальный адресный запуск: 82 passed, 2 CUDA cases skipped в sandbox, 1 unrelated legacy test исключён;
  обе CUDA-проверки ранее отдельно прошли на реальной GPU.
  Regression pipeline parity подтверждён (max error 1.42e-14); CUDA regression/multiclass без numerical
  cpu/numpy/tolist проходят. Проверки graph rules, fork, role ordering, task/target и pickle добавлены.
  Одна unrelated legacy-проверка test_multi_root_pipeline_raise_exception падает на старом ожидании
  текстового сообщения вместо текущего structured validation error; код root validation не менялся.
  **Файлы:** core/pipelines/tensor_input_rules.py, tensor_input.py, node.py, verification_rules.py;
  core/data/merge/data_merger.py; implementations/data_operations/decompose.py;
  strategies/data_operations/common_preprocessing.py; repository/data/data_operation_repository.json;
  core/caching/fingerprints.py, hasher.py (docstring); tests/core/pipelines/test_tensor_input_rules.py;
  tests/core/operations/test_decompose_tensordata.py; test/unit/pipelines/test_pipeline_verification.py.
  Фактический контракт и ограничения:
  [контракты decompose](fedot_tensordata_decompose_contract_proposal_2026_11_11.md).

- [ ] **10. Простые TS-преобразования и базовые прогнозы.**
  Smoothing, Gaussian filter, derivative, cut; модели `locf`, `ts_naive_average`, `polyfit`.
  Сначала закрепить работу с временной осью, горизонтом и выходными индексами на небольших примерах.

- [ ] **11. Lagged/sparse lagged и exogenous TS-преобразования.**
  `ts_transformations.py` + `strategies/data_operations/time_series.py`.
  Train windows, target, idx и последние inference-окна должны соответствовать друг другу;
  проверить связку с уже перенесёнными табличными моделями и согласовать TS-контракт с Industrial.

- [ ] **12. Статистические forecasting-модели.**
  `strategies/models/forecasting.py` + `implementations/models/forecasting/`:
  AR, ARIMA, STL-ARIMA, ETS, GLM. Явно адаптировать NumPy/statsmodels, горизонт и fit-stage predictions.

- [ ] **13. Оставшиеся нейросети для TS/image.**
  CGRU/CLSTM и Keras CNN: TensorData-формы/оси, обучение и предсказание, CPU/GPU средствами библиотек.
  Переиспользовать подходящие общие контракты, не смешивая автоматически с табличной neural-стратегией.

- [ ] **14. Текстовые операции.**
  `strategies/data_operations/text.py` + text implementations:
  cleaning, TF-IDF и embeddings. Проверить представление текста на границе TensorData и схему выхода.

- [x] **15. Топологические признаки — TensorData-контракт.**
  `implementations/data_operations/topological/` и соответствующая стратегия.
  Выполнено вместе с объединением dispatcher 11 ноября: averaging/windows и расширение строк — Torch,
  ripser и Hazen quantiles остаются CPU external boundary. Алгоритм, odd row policy и optional backend сохранены.
  Реальные CPU/GPU boundary результаты совпали с прежней реализацией. Native GPU ripser не добавлен.
  Временный сравнительный скрипт: /tmp/fedot_external_preprocessing_parity.py.

- [ ] **16. Custom callbacks и optional AutoML-обёртки.**
  `models/custom.py`, `strategies/automl.py`, H2O/TPOT wrappers: описать TensorData-контракт границы
  и явные адаптеры внешних библиотек. Современные manifest extensions уже используют TensorData;
  их контракт переиспользовать, а не создавать заново.

- [ ] **17. Итоговая проверка слоя операций.**
  Проверить все зарегистрированные ID, уже готовые neural/cuML/проекции/optional preprocessing/extensions
  и смешанные пайплайны. Удалить оставшиеся ненужные InputData/OutputData bridges и legacy-базы этого слоя,
  когда их потребители перенесены. Отличать неподдерживаемые режимы от незавершённой миграции.

Уже имеющие TensorData-контракт семейства не переписываем заново: neural tabular, новые sklearn/EBM,
cuML, PCA/SVD, optional tabular preprocessing и manifest extensions. Их проверяем при интеграции изменений.

После переноса основных семейств вернуться к общему выбору CPU/GPU по
[чек-листу](fedot_cpu_gpu_operations_checklist_2026_11_07.md) и
[подробному описанию](fedot_cpu_gpu_operations_handoff_2026_11_07.md).
Полноценный TensorData-поиск, native CV/OOF и доработка runtime — отдельные следующие задачи;
перенос операций сам по себе их не завершает.

Инвентарь: [карта моделей и форматов](fedot_operations_data_formats_2026_10_09.md).
Пересечения по idx/merger, TS, cache и capabilities:
[записка Industrial](fedot_industrial_tensordata_sync_2026_11_05.md).
