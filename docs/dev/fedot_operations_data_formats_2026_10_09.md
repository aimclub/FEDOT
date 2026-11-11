# FEDOT: карта моделей, data operations и форматов данных

Дата исходного аудита: 9 октября 2026 года. Имена и пути актуализированы 5 ноября после реструктуризации.
TensorData-only адаптация новых моделей сохранена в `cd281e48`.
Форматы данных остались прежними; Linear/MLP и TabM/RTDL/RealMLP теперь используют общие
`NeuralTabular*Strategy`. Полная карта переезда и уточнения:
[результат реструктуризации](fedot_operations_restructuring_2026_11_05.md).

Уточнение от 6 ноября: старый пакет `evaluation/` удалён, обратных переэкспортов нет.
Старые обученные пайплайны запускаются на исходной версии FEDOT;
совместимость и конверсия сохранённого состояния сейчас не поддерживаются.

Уточнение от 10 ноября: шесть CatBoost/XGBoost/LightGBM wrappers переведены на TensorData-only;
подробности CPU/GPU-границ — в разделе 2.8.
В тот же день все 23 ID прямых sklearn/InterpretML-моделей, включая KMeans объединены под TensorData-контрактом;
временные `TensorSkLearn*` удалены. Следующим шагом в тот же день перенесены FEDOT-адаптеры KNN/SVC/LDA/QDA;
CNN отделён в KerasClassificationStrategy с прежним InputData-контрактом.
Затем TabPFN/TabICL и их четыре GPU-aliases перенесены на TensorData; границы библиотек — в разделе 2.10.

Уточнение от 11 ноября: CPU/GPU-каталоги объединены. Один ID на модель; варианты выбираются по
устройству TensorData, платформе, зависимостям и multi-output и закрепляются при fit. GPU-пресет,
старый GPU JSON и `_gpu` aliases удалены. Текущий статус — в разделах 3 и 7.1.

## 1. Как читать карту

Формат здесь означает контейнер на границе стратегии/реализации, а не библиотеку
алгоритма и не `DataTypesEnum`. `table`/`tabular`, `ts`, `image`, `text` —
типы содержимого; запись `DataTypesEnum.table` в JSON **не доказывает** поддержку InputData.

| Обозначение в таблицах | Значение |
| --- | --- |
| TensorData | TensorData-контракт входа и выхода; InputData-путь отсутствует. |
| InputData | Legacy-контракт InputData → OutputData/NumPy; полноценного TensorData-пути нет. |
| InputData; TensorData CPU (legacy) | InputData-путь сохранён. CPU Torch-тензор может пройти в sklearn через неявное NumPy-преобразование; стратегия может вернуть TensorData с NumPy в `predict`. Это частичная совместимость, а не миграция. |
| Нет рабочего fit-контракта | ID присутствует в каталоге, но текущая стратегия не допускает его обучение. |

Для legacy-строк отметка InputData означает архитектурный контракт реализации,
а не гарантию, что современный публичный `Fedot.fit` и весь pipeline корректно работают
с этим контейнером. Некоторые shell-методы уже ориентированы на TensorData.
Отдельно проверять нужно полный путь: API → pipeline → strategy → implementation → merger.

Для строк с частичной CPU-совместимостью нельзя предполагать поддержку CUDA, нескольких
targets, всех output modes, TS или комбинаций узлов. В исходном аудите на CPU фактически
проверены `logit`, `rf`, `svc`, `knn`, `lda`, `qda`, `kmeans`:
тогда они вернули TensorData с **NumPy ndarray** в `predict`. После миграции sklearn
это историческое наблюдение не относится к уже перенесённым sklearn и табличным адаптерам. Остальные строки этой категории
классифицированы по общему коду стратегии; отдельный runtime-тест каждой не выполнялся.

Внутренний NumPy/Pandas-адаптер у TensorData-only модели допустим. Например, новая
`Sklearn*Strategy` явно переводит Torch → CPU NumPy → Torch на исходном device через общие array_conversions.
Это TensorData-контракт, но не GPU-обучение sklearn. У tabular neural моделей также
есть CPU preprocessing; TensorData-поддержка не означает отсутствие CPU round-trip.

### Каталоги и объём

| Каталог | Число записей | Что покрыто |
| --- | ---: | --- |
| [model_repository.json](../../fedot/core/repository/data/model_repository.json) | 66 | Общие логические модели CPU/GPU; 54 TensorData с вариантами |
| [data_operation_repository.json](../../fedot/core/repository/data/data_operation_repository.json) | 37 | Зарегистрированные data operations после удаления encoder nodes |
| [automl_repository.json](../../fedot/core/repository/data/automl_repository.json) | 4 | H2O/TPOT wrappers |

Всего **107 записей в трёх каталогах**; варианты CPU/GPU находятся внутри одной записи модели.
У 54 TensorData-моделей описаны 66 вариантов исполнения; у 38 моделей есть CUDA-вариант.
Динамические extensions описаны отдельно в разделе 6.

## 2. Основные модели — model_repository.json

### 2.1. Базовые Torch-модели

Стратегии: [NeuralTabularClassificationStrategy](../../fedot/core/operations/strategies/models/neural.py) и [NeuralTabularRegressionStrategy](../../fedot/core/operations/strategies/models/neural.py).
Реализации и локальная настройка обучения: [neural.py](../../fedot/core/operations/implementations/models/neural.py).

Нативные Torch fit/predict; CPU/CUDA. Результат на исходном устройстве TensorData. Tabular features — 2D; один target.

| ID | Стратегия | Реализация | Формат |
| --- | --- | --- | --- |
| `torch_linear` | `NeuralTabularClassificationStrategy` | `TorchLinearClassifier` | TensorData |
| `torch_mlp` | `NeuralTabularClassificationStrategy` | `TorchMLPClassifier` | TensorData |
| `torch_linear_reg` | `NeuralTabularRegressionStrategy` | `TorchLinearRegressor` | TensorData |
| `torch_mlp_reg` | `NeuralTabularRegressionStrategy` | `TorchMLPRegressor` | TensorData |

### 2.2. Дополнительные sklearn/InterpretML-модели

Стратегии: [sklearn.py](../../fedot/core/operations/strategies/models/sklearn.py).
EBM adapters: [estimator_adapters.py](../../fedot/core/operations/implementations/models/estimator_adapters.py).
Нормализация target находится в стратегии; кодирование классов — внутри EBM adapter.

Объединены с остальными sklearn-моделями через `SklearnModelStrategy`; InputData fit/predict-путь удалён.
Вычисления sklearn/InterpretML — CPU. Вход может лежать на CPU/CUDA; результат возвращается
на исходный device. Для regression сохраняется MultiOutputRegressor.
EBM обучается на integer class codes и восстанавливает исходные числовые метки.
JSON по-прежнему разрешает TS для regression ID; это не отдельный нативный TS-контракт.

| ID | Стратегия | Estimator/adapter | Формат |
| --- | --- | --- | --- |
| `extra_trees` | `SklearnClassificationStrategy` | `sklearn.ensemble.ExtraTreesClassifier` | TensorData |
| `hist_gb` | `SklearnClassificationStrategy` | `sklearn.ensemble.HistGradientBoostingClassifier` | TensorData |
| `ebm` | `SklearnClassificationStrategy` | `ExplainableBoostingClassifierAdapter` → InterpretML | TensorData |
| `hist_gbreg` | `SklearnRegressionStrategy` | `sklearn.ensemble.HistGradientBoostingRegressor` | TensorData |
| `ebmreg` | `SklearnRegressionStrategy` | `ExplainableBoostingRegressorAdapter` → InterpretML | TensorData |
| `mlpreg` | `SklearnRegressionStrategy` | `sklearn.neural_network.MLPRegressor` | TensorData |

### 2.3. Tabular neural модели

Стратегии: [neural.py](../../fedot/core/operations/strategies/models/neural.py).
Реализации: [neural.py](../../fedot/core/operations/implementations/models/neural.py).

TensorData-only на границе; CPU/CUDA neural runtime, внутренние NumPy/Pandas-преобразования и CPU preprocessing. Требуются соответствующие внешние библиотеки.

| ID | Стратегия | Реализация | Формат |
| --- | --- | --- | --- |
| `tabm` | `NeuralTabularClassificationStrategy` | `TabMClassifier` | TensorData |
| `tabmreg` | `NeuralTabularRegressionStrategy` | `TabMRegressor` | TensorData |
| `realmlp` | `NeuralTabularClassificationStrategy` | `RealMLPClassifier` | TensorData |
| `realmlpreg` | `NeuralTabularRegressionStrategy` | `RealMLPRegressor` | TensorData |
| `ft_transformer` | `NeuralTabularClassificationStrategy` | `FTTransformerClassifier` | TensorData |
| `ft_transformerreg` | `NeuralTabularRegressionStrategy` | `FTTransformerRegressor` | TensorData |
| `tab_resnet` | `NeuralTabularClassificationStrategy` | `ResNetClassifier` | TensorData |
| `tab_resnetreg` | `NeuralTabularRegressionStrategy` | `ResNetRegressor` | TensorData |

### 2.4. Sklearn classification

Стратегия: [SklearnClassificationStrategy](../../fedot/core/operations/strategies/models/sklearn.py).
Общий fit и output conversion: [SklearnModelStrategy](../../fedot/core/operations/strategies/models/sklearn.py).

TensorData-only. Mapping находится в classification-стратегии; общий fit и конвертеры переиспользуются.
Multioutput trees / MultiOutputClassifier возвращают `(samples, targets, classes)` для full_probs;
MLP multilabel нормализуется в тот же формат. Для binary default/probs остаётся `(samples, targets)`.

| ID | Estimator из sklearn | Формат |
| --- | --- | --- |
| `logit` | `linear_model.LogisticRegression` | TensorData |
| `bernb` | `naive_bayes.BernoulliNB` | TensorData |
| `multinb` | `naive_bayes.MultinomialNB` | TensorData |
| `dt` | `tree.DecisionTreeClassifier` | TensorData |
| `rf` | `ensemble.RandomForestClassifier` | TensorData |
| `mlp` | `neural_network.MLPClassifier` | TensorData |

### 2.5. Sklearn regression

Стратегия: [SklearnRegressionStrategy](../../fedot/core/operations/strategies/models/sklearn.py).
Общий fit: [SklearnModelStrategy](../../fedot/core/operations/strategies/models/sklearn.py).

TensorData-only. JSON допускает regression/TS; алгоритмы sklearn вычисляются на CPU.
Модели с тегом non_multi оборачиваются в MultiOutputRegressor, остальные сохраняют native multioutput.
Одноколоночный target разворачивается в 1D, multioutput сохраняет все столбцы.

| ID | Estimator из sklearn | Формат |
| --- | --- | --- |
| `adareg` | `ensemble.AdaBoostRegressor` | TensorData |
| `gbr` | `ensemble.GradientBoostingRegressor` | TensorData |
| `dtreg` | `tree.DecisionTreeRegressor` | TensorData |
| `treg` | `ensemble.ExtraTreesRegressor` | TensorData |
| `rfr` | `ensemble.RandomForestRegressor` | TensorData |
| `linear` | `linear_model.LinearRegression` | TensorData |
| `ridge` | `linear_model.Ridge` | TensorData |
| `lasso` | `linear_model.Lasso` | TensorData |
| `svr` | `svm.LinearSVR` (не SVR) | TensorData |
| `sgdr` | `linear_model.SGDRegressor` | TensorData |

### 2.6. FEDOT classification adapters и legacy CNN

Стратегия: [AdapterClassificationStrategy](../../fedot/core/operations/strategies/models/adapters.py).

Четыре tabular adapter работают только с TensorData; реализации возвращают Torch, стратегия — TensorData.
Sklearn вычисляет на CPU, переход выполняется общими as_numpy/as_torch только на границе estimator.
Class unique, nan_to_num и probability shaping выполняются в Torch; выход сохраняет device входа.
`cnn` — отдельная KerasClassificationStrategy в strategies/models/keras.py: прежний image InputData-контракт.
Для multilabel KNN/SVC full_probs имеет форму `(samples, targets, classes)`, binary probs — `(samples, targets)`.

| ID | Реализация | Где реализация | Формат |
| --- | --- | --- | --- |
| `lda` | `LDAImplementation` | [estimator_adapters.py](../../fedot/core/operations/implementations/models/estimator_adapters.py) | TensorData |
| `qda` | `QDAImplementation` | [estimator_adapters.py](../../fedot/core/operations/implementations/models/estimator_adapters.py) | TensorData |
| `svc` | `FedotSVCImplementation` (OneVsRestClassifier + SVC) | [estimator_adapters.py](../../fedot/core/operations/implementations/models/estimator_adapters.py) | TensorData |
| `knn` | `KnnClassificationAdapter` | [estimator_adapters.py](../../fedot/core/operations/implementations/models/estimator_adapters.py) | TensorData |
| `cnn` | `FedotCNNImplementation` | [keras.py](../../fedot/core/operations/implementations/models/keras.py) | InputData |

### 2.7. KNN regression и clustering

Стратегии: [AdapterRegressionStrategy](../../fedot/core/operations/strategies/models/adapters.py) и [SklearnClusteringStrategy](../../fedot/core/operations/strategies/models/sklearn.py).

Обе модели принимают TensorData, обучаются в sklearn на CPU NumPy и возвращают Torch на исходный device.
KNN сохраняет коррекцию числа соседей и native target shape; default KMeans n_clusters=2 сохранён.

| ID | Стратегия | Реализация и место | Формат |
| --- | --- | --- | --- |
| `knnreg` | `AdapterRegressionStrategy` | [estimator_adapters.py](../../fedot/core/operations/implementations/models/estimator_adapters.py) | TensorData |
| `kmeans` | `SklearnClusteringStrategy` | `sklearn.cluster.KMeans`, mapping в `strategies/models/sklearn.py` | TensorData |

### 2.8. Boosting

Стратегии: [boosting.py](../../fedot/core/operations/strategies/models/boosting.py).
Все FEDOT wrappers: [boosting.py](../../fedot/core/operations/implementations/models/boosting.py).

TensorData-only. Реализации возвращают Torch predictions на устройстве входных данных,
стратегии записывают их в TensorData.predict. Общая подготовка — `BoostingImplementation`.
Split сохраняет политику стратификации; теперь она выполняется Torch, поэтому конкретные RNG-строки отличаются.
Eval/early stopping и multioutput сохранены, последовательный default общего split не изменён.
При `enable_categorical=True` берутся исходные категории с учётом optional imputation.
В boosting-коде pandas отсутствует.

XGBoost 3.0.5 получает Torch features/target/eval напрямую, CPU/CUDA; явный NumPy/CuPy adapter удалён.
CPU-вход внутри XGBoost читается через NumPy, CUDA-вход — через CUDA Array Interface.
CUDA predict выполняется через Booster.inplace_predict → CuPy → Torch DLPack без NumPy roundtrip.
Сохранены best iteration, multiclass/multilabel/softmax; CPU и gblinear используют sklearn-путь.
LightGBM принимает NumPy; CatBoost — Pool/FeaturesData с отдельными числовым и строковым блоками.
Для них CUDA features проходят через host API, хотя само обучение выполняется на GPU при явной настройке.
Проверены CPU и реальные GPU-запуски всех трёх семейств, включая исходные категории с пропусками.
Автоматический выбор устройства и объединение CPU/GPU-каталогов остаются отдельной задачей.

| ID | Стратегия | Реализация | Формат |
| --- | --- | --- | --- |
| `xgboost` | `BoostingClassificationStrategy` | `FedotXGBoostClassificationImplementation` | TensorData |
| `xgboostreg` | `BoostingRegressionStrategy` | `FedotXGBoostRegressionImplementation` | TensorData |
| `lgbm` | `BoostingClassificationStrategy` | `FedotLightGBMClassificationImplementation` | TensorData |
| `lgbmreg` | `BoostingRegressionStrategy` | `FedotLightGBMRegressionImplementation` | TensorData |
| `catboost` | `BoostingClassificationStrategy` | `FedotCatBoostClassificationImplementation` | TensorData |
| `catboostreg` | `BoostingRegressionStrategy` | `FedotCatBoostRegressionImplementation` | TensorData |

### 2.9. Time series models

Стратегия для всех строк: [ForecastingStrategy](../../fedot/core/operations/strategies/models/forecasting.py).
Файлы реализации находятся в `fedot/core/operations/evaluation/operation_implementations/models/ts_implementations/`.

Legacy TS-контракт: NumPy/SciPy/statsmodels, forecast_length и OutputData.
Даже `cgru`/`clstm`, обучающиеся через Torch, читают NumPy InputData и создают tensors
внутри wrapper. Они не относятся к TensorData-native моделям.

| ID | Реализация | Файл | Формат |
| --- | --- | --- | --- |
| `ar` | `AutoRegImplementation` | [statsmodels.py](../../fedot/core/operations/implementations/models/forecasting/statsmodels.py) | InputData |
| `arima` | `ARIMAImplementation` | [arima.py](../../fedot/core/operations/implementations/models/forecasting/arima.py) | InputData |
| `stl_arima` | `STLForecastARIMAImplementation` | [arima.py](../../fedot/core/operations/implementations/models/forecasting/arima.py) | InputData |
| `glm` | `GLMImplementation` | [statsmodels.py](../../fedot/core/operations/implementations/models/forecasting/statsmodels.py) | InputData |
| `ets` | `ExpSmoothingImplementation` | [statsmodels.py](../../fedot/core/operations/implementations/models/forecasting/statsmodels.py) | InputData |
| `polyfit` | `PolyfitImplementation` | [poly.py](../../fedot/core/operations/implementations/models/forecasting/poly.py) | InputData |
| `locf` | `RepeatLastValueImplementation` | [naive.py](../../fedot/core/operations/implementations/models/forecasting/naive.py) | InputData |
| `ts_naive_average` | `NaiveAverageForecastImplementation` | [naive.py](../../fedot/core/operations/implementations/models/forecasting/naive.py) | InputData |
| `cgru` | `CGRUImplementation` | [cgru.py](../../fedot/core/operations/implementations/models/forecasting/cgru.py) | InputData |
| `clstm` | `CLSTMImplementation` | [cgru.py](../../fedot/core/operations/implementations/models/forecasting/cgru.py) | InputData |

### 2.10. Foundational models

Стратегии: [foundation.py](../../fedot/core/operations/strategies/models/foundation.py).
Реализации: [foundation.py](../../fedot/core/operations/implementations/models/foundation.py).

TensorData-only для CPU/GPU. BaseFoundationalImplementation возвращает Torch prediction на device входа,
стратегия — TensorData с исходными features/metadata. Контейнерных bridges и OutputData внутри семейства нет.

| Библиотека / установленная версия | Features и target в public API | Выход public API | Что делает FEDOT |
| --- | --- | --- | --- |
| TabPFN 2.2.1, classification/regression | CPU NumPy; при raw categories — mixed object NumPy | NumPy labels / probabilities / regression mean | as_numpy перед библиотекой; as_torch на исходный device после неё |
| TabICL 2.1.1, classification/regression | CPU NumPy, собственный sklearn preprocessing | NumPy labels / probabilities / regression | Те же общие конвертеры; transformer может вычислять на CUDA |

Преобразование — требование обычного library preprocessing. В FEDOT unique классов и target reshape — Torch.
Differentiable-input TabPFN не включён: он меняет transforms/ensembles; внутренний forward не подменяет
public predict ради обхода NumPy, поскольку это потребовало бы дублировать библиотечную постобработку.

TabPFN enable_categorical использует get_not_encoded_data, сохраняя обработку категорий самой библиотекой.
Категориальные индексы задаются по восстановленному порядку numerical + categories, а не по OHE-позициям.
Отсутствие optional imputation сохраняет missing values; unknown categories поступают native encoder.
TabICL продолжает использовать уже закодированные TensorData.features. Лимиты FEDOT строк/признаков
и параметры устройства сохранены; превышение вызывает ошибку, не скрытое сокращение данных.

| ID | Стратегия | Реализация | Формат |
| --- | --- | --- | --- |
| `tabpfn` | `TabPFNClassificationStrategy` | `FedotTabPFNClassificationImplementation` | TensorData |
| `tabpfnreg` | `TabPFNRegressionStrategy` | `FedotTabPFNRegressionImplementation` | TensorData |
| `tabicl` | `TabICLClassificationStrategy` | `FedotTabICLClassificationImplementation` | TensorData |
| `tabiclreg` | `TabICLRegressionStrategy` | `FedotTabICLRegressionImplementation` | TensorData |

### 2.11. Legacy custom model

Стратегия: [CustomModelStrategy](../../fedot/core/operations/strategies/models/custom.py).
Реализация: [CustomModelImplementation](../../fedot/core/operations/implementations/models/custom.py).

Callbacks получают idx/features/target/params, но wrapper возвращает OutputData
через legacy converter. Произвольный callback не делает wrapper TensorData-native.
Современные manifest extensions — другой механизм, см. раздел 6.

| ID | Реализация | Формат |
| --- | --- | --- |
| `custom` | `CustomModelImplementation` | InputData |

## 3. GPU-варианты общего model_repository.json

Одна логическая модель может иметь sklearn CPU и cuML CUDA варианты. CUDA-вход выбирает допустимый
CUDA-вариант, иначе CPU; CPU-вход выбирает CPU. `cd`, `elasticnet`, `mbsgdcregr`, `minibatchsgd` пока
имеют только cuML-вариант и недоступны для CPU-входа. Неработающий low-level `sgd` alias удалён.

Общая cuML база: [cuml.py](../../fedot/core/operations/strategies/models/cuml.py).
Адаптация параметров, выбор dtype и проверка окружения находятся в общей стратегии.
Mapping estimator-классов расположен в `strategies/models/cuml.py`, а не в отдельных model wrappers.

После последнего перевода cuML принимает только TensorData. CUDA Torch → CuPy → cuML
и CuPy → Torch используют DLPack. При fit selector направляет CPU-вход в CPU-вариант;
при predict уже обученный cuML estimator использует закреплённую CUDA, даже если новые данные на CPU,
а результат возвращается на устройство входа. Требуются native Linux, RAPIDS и CUDA;
WSL текущая проверка окружения отклоняет.

### 3.1. cuML classification

Стратегия: [CuMLClassificationStrategy](../../fedot/core/operations/strategies/models/cuml.py).

`minibatchsgd` не обещает predict_proba: labels доступны, explicit probs/full_probs могут
быть отклонены. Default без predict_proba использует labels.

| ID | cuML estimator | Формат |
| --- | --- | --- |
| `logit` | `cuml.linear_model.LogisticRegression` | TensorData |
| `rf` | `cuml.ensemble.RandomForestClassifier` | TensorData |
| `svc` | `cuml.svm.SVC` | TensorData |
| `knn` | `cuml.neighbors.KNeighborsClassifier` | TensorData |
| `multinb` | `cuml.naive_bayes.MultinomialNB` | TensorData |
| `bernb` | `cuml.naive_bayes.BernoulliNB` | TensorData |
| `minibatchsgd` | `cuml.linear_model.MBSGDClassifier` | TensorData |

### 3.2. cuML regression

Стратегия: [CuMLRegressionStrategy](../../fedot/core/operations/strategies/models/cuml.py).

Один target; `cd` в каталоге и стратегии относится к regression.

| ID | cuML estimator | Формат |
| --- | --- | --- |
| `linear` | `cuml.linear_model.LinearRegression` | TensorData |
| `ridge` | `cuml.linear_model.Ridge` | TensorData |
| `lasso` | `cuml.linear_model.Lasso` | TensorData |
| `elasticnet` | `cuml.linear_model.ElasticNet` | TensorData |
| `rfr` | `cuml.ensemble.RandomForestRegressor` | TensorData |
| `knnreg` | `cuml.neighbors.KNeighborsRegressor` | TensorData |
| `mbsgdcregr` | `cuml.linear_model.MBSGDRegressor` | TensorData |
| `cd` | `cuml.solvers.CD` | TensorData |

### 3.3. cuML clustering

Стратегия: [CuMLClusteringStrategy](../../fedot/core/operations/strategies/models/cuml.py) (именно такое написание класса).

Target не требуется.

| ID | cuML estimator | Формат |
| --- | --- | --- |
| `kmeans` | `cuml.cluster.KMeans` | TensorData |

### 3.4. Foundational CPU/GPU под общими ID

Те же [foundational strategies](../../fedot/core/operations/strategies/models/foundation.py) и [foundational implementations](../../fedot/core/operations/implementations/models/foundation.py), что в разделе 2.10.

У этих моделей один вариант с devices=[cpu,cuda], это не cuML. Стратегия передаёт конкретное device.
max_samples=None выбирает прежний cap: TabPFN CPU/CUDA 1000/10000, TabICL 5000/30000;
явный числовой override сохраняется. GPU aliases удалены.

| ID | Стратегия | Реализация | Формат |
| --- | --- | --- | --- |
| `tabpfn` | `TabPFNClassificationStrategy` | `FedotTabPFNClassificationImplementation` | TensorData |
| `tabpfnreg` | `TabPFNRegressionStrategy` | `FedotTabPFNRegressionImplementation` | TensorData |
| `tabicl` | `TabICLClassificationStrategy` | `FedotTabICLClassificationImplementation` | TensorData |
| `tabiclreg` | `TabICLRegressionStrategy` | `FedotTabICLRegressionImplementation` | TensorData |

## 4. Data operations — data_operation_repository.json

Обновлено 11 ноября: general preprocessing и проекции объединены. OHE/label encoding pipeline nodes
удалены, их функции выполняет obligatory TensorData preparation. Tabular imputation выполняет
optional_preprocessing; InputData simple_imputation оставлен только для отложенного TS-пути.

### 4.1. TensorData-native transforms и optional preprocessing

Стратегии: [PreprocessingDataOperationStrategy](../../fedot/core/operations/strategies/data_operations/common_preprocessing.py) и [OptionalPreprocessingStrategy](../../fedot/core/operations/strategies/data_operations/optional.py).

PCA/SVD сохраняют Torch runtime и возвращают transformed `features`, а не model `predict`.
Для optional preprocessing в core зарегистрирован **только tabular runtime**:
[tensor_optional_runtime.py](../../fedot/preprocessing/service/tensor_optional_runtime.py).
TS/text/image runtime должен быть зарегистрирован extension; широкие JSON tasks/input_types
не доказывают наличие такого runtime в core.

| ID | Стратегия | Реализация и файл | Формат |
| --- | --- | --- | --- |
| `pca` | `PreprocessingDataOperationStrategy` | [features_reducing.py](../../fedot/core/operations/implementations/data_operations/features_reducing.py) | TensorData |
| `truncated_svd` | `PreprocessingDataOperationStrategy` | [features_reducing.py](../../fedot/core/operations/implementations/data_operations/features_reducing.py) | TensorData |
| `optional_preprocessing` | `OptionalPreprocessingStrategy` | [OptionalService](../../fedot/preprocessing/service/optional_service.py) / [OptionalTabularService](../../fedot/preprocessing/service/tabular_optional_service.py) | TensorData |

### 4.2. Общие преобразования TensorData

Стратегия для всех строк — PreprocessingDataOperationStrategy из common_preprocessing.py.
Алгоритмы native Torch и CPU libraries разделены в implementations, без отдельных стратегий по формату.

| ID | Реализация | Где находится | Формат / вычисления |
| --- | --- | --- | --- |
| scaling | ScalingImplementation | implementations/data_operations/tabular.py | TensorData, native Torch CPU/CUDA |
| normalization | NormalizationImplementation | implementations/data_operations/tabular.py | TensorData, native Torch CPU/CUDA, min-max |
| poly_features | PolyFeaturesImplementation | implementations/data_operations/tabular.py | TensorData, native Torch CPU/CUDA |
| kernel_pca | KernelPCAImplementation | implementations/data_operations/features_reducing.py | TensorData, CPU sklearn boundary |
| fast_ica | FastICAImplementation | implementations/data_operations/features_reducing.py | TensorData, CPU sklearn boundary |
| dask_pca | DaskPCAImplementation | implementations/data_operations/features_reducing.py | TensorData, CPU Dask boundary |
| topological_features | TopologicalFeaturesImplementation | implementations/data_operations/topological/fast_topological_extractor.py | TensorData; Torch windows/rows, CPU ripser/quantiles |

Scaling/normalization сохраняют исходный порядок и пропускают fitted columns с <=2 unique values;
label-коды с большим количеством значений масштабируются. get_not_encoded_data восстанавливает raw categories,
сохраняя текущие numerical columns. Статистики scaler общие с optional preprocessing, поддержаны clip и
перенос fitted state на устройство inference. Прежний nan_to_num на выходе узлов сохранён.

Legacy simple_imputation (ImputationImplementation в tabular.py) остался InputData-only для TS/Industrial;
из табличного каталога задач он исключён. OptionalPreprocessingStrategy разрешён после промежуточных нод,
а не только на входе; исходный TensorData при imputation/scaling не изменяется.

### 4.2.1. Отбор признаков TensorData

Стратегия: PreprocessingDataOperationStrategy (common_preprocessing.py).
Реализации: [sklearn_selectors.py](../../fedot/core/operations/implementations/data_operations/sklearn_selectors.py).

| ID | Базовая sklearn-модель | Формат / вычисления |
| --- | --- | --- |
| rfe_lin_class | LogisticRegression | TensorData; RFE.fit CPU NumPy, transform Torch CPU/CUDA |
| rfe_non_lin_class | DecisionTreeClassifier | TensorData; RFE.fit CPU NumPy, transform Torch CPU/CUDA |
| rfe_lin_reg | LinearRegression | TensorData; RFE.fit CPU NumPy, transform Torch CPU/CUDA |
| rfe_non_lin_reg | DecisionTreeRegressor | TensorData; RFE.fit CPU NumPy, transform Torch CPU/CUDA |

Все четыре используют TensorData.subset_features: схема колонок и categorical sidecar обновляются вместе.
Частично оставленный OHE-блок сохраняет исходный category source, полностью удалённый не восстанавливается.
GPU-обучение RFE отложено; наличие CUDA tensor на входе не означает GPU fit. InputData-путь удалён.

### 4.2.2. Фильтры и балансировка TensorData

Стратегия: PreprocessingDataOperationStrategy (common_preprocessing.py).
Изменяют строки только в transform_for_fit; обычный transform возвращает все входные строки.

| ID | Реализация и файл | Формат / вычисления |
| --- | --- | --- |
| ransac_lin_reg | LinearRegRANSACImplementation, sklearn_filters.py | TensorData; sklearn CPU fit, Torch thresholds/mask/rows |
| ransac_non_lin_reg | NonLinearRegRANSACImplementation, sklearn_filters.py | TensorData; sklearn CPU fit, Torch thresholds/mask/rows |
| isolation_forest_reg | IsolationForestRegImplementation, sklearn_filters.py | TensorData; sklearn CPU fit/predict, Torch mask/rows |
| isolation_forest_class | IsolationForestClassImplementation, sklearn_filters.py | То же + Torch class retention guard |
| resample | ResampleImplementation, resampling.py | Native Torch CPU/CUDA, NumPy/sklearn в runtime нет |

TensorData.subset_rows сохраняет согласованность tensor fields, labels и original categories.
Resample сохраняет прежние ratio/replacement/order, после балансировки выдаёт новые positional idx;
Torch RNG отличается от NumPy RNG. Одинаковые random draws дают тот же результат, target dtype/shape
сохраняются из TensorData. GPU fit filters отложен; CUDA resample проверен без CPU-переходов чисел.

### 4.3. Регрессионный decompose

Стратегия: PreprocessingDataOperationStrategy (common_preprocessing.py).
Реализация: DecomposerRegImplementation в implementations/data_operations/decompose.py.
Формат: TensorData, native Torch CPU/CUDA. Fit-stage target = y - baseline, inference сохраняет признаки.
Входные роли и источник цели разрешаются по графу в pipelines/tensor_input_rules.py / tensor_input.py.
Новых supplementary/target-role/depth fields в TensorData нет. Старые task-specific стратегии удалены.

### 4.4. Классификационный decompose

Стратегия: тот же PreprocessingDataOperationStrategy.
Реализация: DecomposerClassImplementation в implementations/data_operations/decompose.py.
Формат: TensorData, native Torch CPU/CUDA; class mapping, binary/multiclass residuals.
После операции task=regression. При последующем main merge исходная задача и target выбираются из
основной ветви; ensemble внутри residual fork продолжает учиться на локальных остатках.
Legacy binary first-row argwhere defect исправлен после обсуждения с пользователем — см.
[контракт](fedot_tensordata_decompose_contract_proposal_2026_11_11.md).
Общий multioutput multiclass/ragged probability contract здесь не вводится.

### 4.5. Legacy time series transforms

Стратегия для всех строк: [TimeSeriesDataOperationStrategy](../../fedot/core/operations/strategies/data_operations/time_series.py).
Все реализации: [ts_transformations.py](../../fedot/core/operations/implementations/data_operations/ts_transformations.py).

Legacy TS/forecast layout, NumPy/Pandas, OutputData и supplementary column metadata. Native TensorData TS metadata/row-changing contract отсутствует.

| ID | Реализация | Формат |
| --- | --- | --- |
| `lagged` | `LaggedTransformationImplementation` | InputData |
| `sparse_lagged` | `SparseLaggedTransformationImplementation` | InputData |
| `smoothing` | `TsSmoothingImplementation` | InputData |
| `gaussian_filter` | `GaussianFilterImplementation` | InputData |
| `diff_filter` | `NumericalDerivativeFilterImplementation` | InputData |
| `cut` | `CutImplementation` | InputData |
| `exog_ts` | `ExogDataTransformationImplementation` | InputData |

### 4.6. Legacy text transforms

Стратегии и vectorizer mapping: [text.py](../../fedot/core/operations/strategies/data_operations/text.py).

Raw strings/string arrays и сторонние текстовые библиотеки; TensorData с numeric features не является этим входным контрактом.

| ID | Стратегия | Реализация и место | Формат |
| --- | --- | --- | --- |
| `text_clean` | `TextCleaningStrategy` | [text_preprocessing.py](../../fedot/core/operations/implementations/data_operations/text_preprocessing.py) | InputData |
| `cntvect` | `SklearnTextVectorizationStrategy` | `sklearn.feature_extraction.text.CountVectorizer`, mapping в `text.py` | InputData |
| `tfidf` | `SklearnTextVectorizationStrategy` | `sklearn.feature_extraction.text.TfidfVectorizer`, mapping в `text.py` | InputData |
| `word2vec_pretrained` | `PretrainedEmbeddingStrategy` | [text_pretrained.py](../../fedot/core/operations/implementations/data_operations/text_pretrained.py) | InputData |

### 4.7. Data sources

Стратегия и реализация для всех строк: [DataSourceStrategy](../../fedot/core/operations/strategies/sources.py); отдельного implementation-класса нет.

Fit возвращает object(); predict возвращает TensorData с теми же features и очищенным predict.
NumPy/OutputData-конверсий нет. Это перенос passthrough-контракта; полноценные raw text/image/TS runtime
и multimodal API ещё требуют своих последующих этапов.

| ID | Назначение | Формат |
| --- | --- | --- |
| `data_source_table` | Tabular source | TensorData |
| `data_source_ts` | TS source | TensorData |
| `data_source_img` | Image source | TensorData |
| `data_source_text` | Text source | TensorData |

## 5. AutoML wrappers — automl_repository.json

Все четыре стратегии находятся в [automl.py](../../fedot/core/operations/strategies/automl.py).
Estimator-классы — из H2O/TPOT; serialization wrappers находятся в
[automl_wrappers.py](../../fedot/core/pipelines/automl_wrappers.py).
Это отдельный optional repository, не основной model mapping.

| ID | Стратегия | Estimator / wrapper | Формат |
| --- | --- | --- | --- |
| `h2o_regr` | `H2OAutoMLRegressionStrategy` | H2OAutoML → H2OSerializationWrapper | InputData |
| `h2o_class` | `H2OAutoMLClassificationStrategy` | H2OAutoML → H2OSerializationWrapper | InputData |
| `tpot_regr` | `TPOTAutoMLRegressionStrategy` | TPOTRegressor → TPOTRegressionSerializationWrapper | InputData |
| `tpot_class` | `TPOTAutoMLClassificationStrategy` | TPOTClassifier → fitted sklearn pipeline | InputData |

H2O создаёт H2OFrame и инициализирует сервер. TPOT использует NumPy `.astype()`.
Нативных TensorData adapters нет. Доступность актуальных H2O/TPOT зависимостей и
работоспособность legacy-кода в этом аудите не проверялись; H2O server не запускался.

## 6. Динамические extensions, не фиксированные ID из JSON

| Вид | Стратегия | Формат на границе FEDOT | Где реализация |
| --- | --- | --- | --- |
| Manifest model | `ExtensionModelStrategy` | TensorData | [extensions.py](../../fedot/core/operations/strategies/extensions.py), [ExtensionModel](../../fedot/core/operations/extension_model.py) |
| Manifest transform | `ExtensionTransformStrategy` | TensorData | [extensions.py](../../fedot/core/operations/strategies/extensions.py), [ExtensionTransform](../../fedot/core/operations/extension_transform.py) |

ID, factories, задачи, типы данных и backend определяются зарегистрированными
`ExternalModelSpec`/`ExternalTransformSpec`, поэтому перечислить их как постоянный
набор встроенных моделей нельзя. Registry: [fedot/extensions/registry.py](../../fedot/extensions/registry.py).
Модельный callback получает `ModelInput`, transformer — `TransformInput`;
массивы могут быть NumPy или Torch по capabilities. Adapter возвращает Torch fields
в TensorData на исходном device. Это **не** legacy `custom` ID из раздела 2.11.

## 7. Где остаются препятствия миграции

### 7.1. Сводка именно по operation runtime

Обновлено 11 ноября после переноса decompose и объединения CPU/GPU. Это число записей каталога с контейнерным
TensorData-контрактом, а не число native GPU algorithms и не результат полного search acceptance.

| Каталог | TensorData-контракт | Остальное |
| --- | ---: | --- |
| Общие модели CPU/GPU | 54 | 10 TS-моделей, Keras CNN и legacy custom wrapper остаются InputData-oriented |
| Data operations | 25 | 7 TS transforms, legacy TS imputation и 4 текстовые операции |
| AutoML | 0 | 4 wrappers H2O/TPOT остаются InputData-oriented |

Основные tabular classification/regression семейства и KMeans уже перенесены. CPU sklearn/library
boundaries сохраняются там, где это необходимо. GPU-варианты объединены; обычные пресеты сохранены,
доступность дополнительно фильтруется перед поиском. Execution-зависимые пространства тюнинга,
предпочтения и отложенные TS/image/text семейства остаются отдельными задачами.

### 7.2. Общие границы, которые объясняют статусы

- [EvaluationStrategy](../../fedot/core/operations/strategies/base.py):
  общий `_convert_to_output` умеет прикрепить NumPy prediction к TensorData.
  Поэтому одинаковый тип внешнего контейнера не доказывает перенос алгоритма.
- [base.py](../../fedot/core/operations/implementations/base.py):
  legacy implementation converter читает `encoded_idx`, `categorical_features`,
  `supplementary_data`; это прямой blocker для большинства legacy implementations.
- [Operation](../../fedot/core/operations/operation.py):
  TensorData shell переупаковывает некоторые raw/OutputData результаты, но не исправляет
  ошибки, произошедшие внутри реализации до возврата результата.
- [PipelineNode](../../fedot/core/pipelines/node.py) и
  [TensorDataMerger](../../fedot/core/data/merge/data_merger.py):
  model parents передают `predict`, transform parents — `features`.
  Legacy transforms должны переносить schema, device и row metadata; смены аннотации недостаточно.
- [tensor_optional_runtime.py](../../fedot/preprocessing/service/tensor_optional_runtime.py):
  встроенная регистрация optional runtime только tabular; TS/text/image требуют extension.
- [CuMLModelStrategy](../../fedot/core/operations/strategies/models/cuml.py):
  напрямую проверяет наличие cuML, native Linux и CUDA. Общего selector движков нет.
  Модельные планы/rules для sklearn, Torch и cuML удалены 9 октября:
  простая адаптация параметров и детали алгоритмов остаются локальными.

### 7.3. Практический список следующих шагов

1. Основные tabular модели и data operations готовы к этапу унификации CPU/GPU: общий каталог,
   выбор варианта до стратегии, пользовательское управление устройствами, сохранение выбранного
   варианта после fit. Предпочтения и добавление новых GPU алгоритмов — отдельные решения.
2. Проверить совместимый пул tabular операций в initial assumptions / mutations / evaluators;
   перенос runtime операции ещё не означает полноценный TensorData evolutionary search.
3. TS models/transforms, text/image и multimodal остаются отдельными acceptance gates.
   TS по договорённости отложены, это не препятствует общей CPU/GPU-инфраструктуре для tabular.
4. Legacy custom callback и H2O/TPOT wrappers требуют отдельного контракта; manifest extensions
   уже используют современный TensorData interface.
5. Native CV/OOF, sampling/chunks, remote evaluation и auxiliary API лежат за пределами переноса
   операций. После CPU/GPU-этапа продолжить завершение полного TensorData search/runtime.

Подробный контекст search и последовательность этапов:
[fedot_tensordata_handoff_2026_10_08.md](fedot_tensordata_handoff_2026_10_08.md)
и [tensordata_full_pipeline_search_plan.md](tensordata_full_pipeline_search_plan.md).

## 8. Как проверялось и как обновлять

Основа аудита — JSON metadata → конкретная strategy → estimator/implementation →
output converter. Исходный аудит охватывал 126 записей; после удаления двух legacy encoder nodes
в четырёх каталогах тогда было 124 записи (историческое число; сейчас 107 в трёх). Таблицы runtime обновлялись при переносе семейств; текущая сводка
на 11 ноября приведена в разделе 7.1. Формат не определялся только по type hints.

В этом проходе дополнительно выполнены CPU smoke checks для семи legacy моделей и
`data_source_table`; ошибки `scaling` (`.iloc` у Torch tensor) и `resample`
(`encoded_idx` у TensorData) воспроизведены. CUDA, foundational downloads и
H2O/TPOT обучение здесь не запускались.

Предыдущая задача проверила новые модельные контракты, включая CUDA/cuML:
111 тестов в общем запуске и отдельный regression test EBM numeric labels (всего 112).
Это относится к предыдущей задаче, а не к полному проверочному запуску всех operations.

После удаления избыточных модельных plans/rules: 112 тестов прошли, 27 GPU-тестов
пропущены. Проверены модельные контракты, EBM numeric labels/serialization/random state,
Torch configuration, TensorData merge/bridge, OOF и chunked reuse. Повторная проверка
реального CUDA/cuML runtime не состоялась: вне sandbox `nvidia-smi` сообщает
`Driver/library version mismatch`, а cuDF import — `cudaErrorCompatNotSupportedOnDevice`.

Основные тесты подтверждённых TensorData путей:

После реструктуризации: `tests/core/operations` — 183 passed, 25 skipped.
Проверены режимы нейросетевых семейств и сохранение/загрузка новых пайплайнов;
расширенный прогон и его baseline-сравнение описаны в [отчёте](fedot_operations_restructuring_2026_11_05.md).

- [test_torch_models.py](../../tests/core/operations/test_torch_models.py).
- [test_experimental_tabular_models.py](../../tests/core/operations/test_experimental_tabular_models.py).
- [test_torch_model_config.py](../../tests/core/operations/test_torch_model_config.py).
- [test_cuml_strategy.py](../../tests/core/operations/test_cuml_strategy.py).
- [test_cuml_tensordata.py](../../tests/core/operations/test_cuml_tensordata.py).
- [test_features_reducing.py](../../tests/core/operations/test_features_reducing.py).
- [test_extension_tensor_pipeline.py](../../tests/core/operations/test_extension_tensor_pipeline.py).

При изменении стратегии обновлять здесь и format, и путь реализации, и ограничения.
Если появляется только CPU smoke compatibility, не заменять её отметкой полноценного
TensorData-контракта без device/schema/output проверки.

**Сохранение в Git:** карта включена в коммит реструктуризации явным добавлением,
поскольку правило `*.md` в `.gitignore` скрывает новые Markdown-файлы.
