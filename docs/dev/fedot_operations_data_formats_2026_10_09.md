# FEDOT: карта моделей, data operations и форматов данных

Дата исходного аудита: 9 октября 2026 года. Имена и пути актуализированы 5 ноября после реструктуризации.
TensorData-only адаптация новых моделей сохранена в `cd281e48`.
Форматы данных остались прежними; Linear/MLP и TabM/RTDL/RealMLP теперь используют общие
`NeuralTabular*Strategy`. Полная карта переезда и уточнения:
[результат реструктуризации](fedot_operations_restructuring_2026_11_05.md).

Уточнение от 6 ноября: старый пакет `evaluation/` удалён, обратных переэкспортов нет.
Старые обученные пайплайны запускаются на исходной версии FEDOT;
совместимость и конверсия сохранённого состояния сейчас не поддерживаются.

## 1. Как читать карту

Формат здесь означает контейнер на границе стратегии/реализации, а не библиотеку
алгоритма и не `DataTypesEnum`. `table`/`tabular`, `ts`, `image`, `text` —
типы содержимого; запись `DataTypesEnum.table` в JSON **не доказывает** поддержку InputData.

| Обозначение в таблицах | Значение |
| --- | --- |
| TensorData | TensorData-контракт входа и выхода; InputData-путь отсутствует. |
| InputData | Legacy-контракт InputData → OutputData/NumPy; полноценного TensorData-пути нет. |
| InputData; TensorData CPU (legacy) | InputData-путь сохранён. CPU Torch-тензор может пройти в sklearn через неявное NumPy-преобразование; стратегия может вернуть TensorData с NumPy в `predict`. Это частичная совместимость, а не миграция. |
| InputData; TensorData через OutputData | Legacy passthrough-стратегия возвращает OutputData; TensorData shell извлекает результат и возвращает TensorData. |
| Нет рабочего fit-контракта | ID присутствует в каталоге, но текущая стратегия не допускает его обучение. |

Для legacy-строк отметка InputData означает архитектурный контракт реализации,
а не гарантию, что современный публичный `Fedot.fit` и весь pipeline корректно работают
с этим контейнером. Некоторые shell-методы уже ориентированы на TensorData.
Отдельно проверять нужно полный путь: API → pipeline → strategy → implementation → merger.

Для строк с частичной CPU-совместимостью нельзя предполагать поддержку CUDA, нескольких
targets, всех output modes, TS или комбинаций узлов. В этом аудите на CPU фактически
проверены `logit`, `rf`, `svc`, `knn`, `lda`, `qda`, `kmeans`:
они вернули TensorData с **NumPy ndarray** в `predict`. Остальные строки этой категории
классифицированы по общему коду стратегии; отдельный runtime-тест каждой не выполнялся.

Внутренний NumPy/Pandas-адаптер у TensorData-only модели допустим. Например, новая
`TensorSkLearn*Strategy` явно переводит Torch → CPU NumPy → Torch на исходном device.
Это TensorData-контракт, но не GPU-обучение sklearn. У tabular neural моделей также
есть CPU preprocessing; TensorData-поддержка не означает отсутствие CPU round-trip.

### Каталоги и объём

| Каталог | Число записей | Что покрыто |
| --- | ---: | --- |
| [model_repository.json](../../fedot/core/repository/data/model_repository.json) | 62 | Основные модели |
| [gpu_models_repository.json](../../fedot/core/repository/data/gpu_models_repository.json) | 21 | cuML, GPU-алиасы foundational моделей и неработающий solver alias |
| [data_operation_repository.json](../../fedot/core/repository/data/data_operation_repository.json) | 39 | Все зарегистрированные data operations |
| [automl_repository.json](../../fedot/core/repository/data/automl_repository.json) | 4 | H2O/TPOT wrappers |

Всего **126 записей каталогов**, не 126 уникальных алгоритмов: ID вроде `rf`, `linear`,
`kmeans` повторяются в обычном и GPU-каталогах с разными стратегиями.
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

### 2.2. Новые sklearn/InterpretML-модели

Стратегии: [sklearn.py](../../fedot/core/operations/strategies/models/sklearn.py).
EBM adapters: [estimator_adapters.py](../../fedot/core/operations/implementations/models/estimator_adapters.py).
Нормализация target находится в стратегии; кодирование классов — внутри EBM adapter.

Отделены от `SklearnModelStrategy`; InputData fit/predict-путь удалён.
Вычисления sklearn/InterpretML — CPU. Вход может лежать на CPU/CUDA; результат возвращается
на исходный device. Для regression сохраняется MultiOutputRegressor.
EBM обучается на integer class codes и восстанавливает исходные числовые метки.
JSON по-прежнему разрешает TS для regression ID; это не отдельный нативный TS-контракт.

| ID | Стратегия | Estimator/adapter | Формат |
| --- | --- | --- | --- |
| `extra_trees` | `TensorSkLearnClassificationStrategy` | `sklearn.ensemble.ExtraTreesClassifier` | TensorData |
| `hist_gb` | `TensorSkLearnClassificationStrategy` | `sklearn.ensemble.HistGradientBoostingClassifier` | TensorData |
| `ebm` | `TensorSkLearnClassificationStrategy` | `ExplainableBoostingClassifierAdapter` → InterpretML | TensorData |
| `hist_gbreg` | `TensorSkLearnRegressionStrategy` | `sklearn.ensemble.HistGradientBoostingRegressor` | TensorData |
| `ebmreg` | `TensorSkLearnRegressionStrategy` | `ExplainableBoostingRegressorAdapter` → InterpretML | TensorData |
| `mlpreg` | `TensorSkLearnRegressionStrategy` | `sklearn.neural_network.MLPRegressor` | TensorData |

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

### 2.4. Legacy sklearn classification

Стратегия: [SklearnClassificationStrategy](../../fedot/core/operations/strategies/models/sklearn.py).
Общий fit, estimator mapping и output conversion: [SklearnModelStrategy](../../fedot/core/operations/strategies/models/sklearn.py).

Это оставшийся legacy-пул, не затронутый переводом шести новых моделей. TensorData CPU — частичный путь по общей стратегии, без явного device adapter.

| ID | Estimator из sklearn | Формат |
| --- | --- | --- |
| `logit` | `linear_model.LogisticRegression` | InputData; TensorData CPU (legacy) |
| `bernb` | `naive_bayes.BernoulliNB` | InputData; TensorData CPU (legacy) |
| `multinb` | `naive_bayes.MultinomialNB` | InputData; TensorData CPU (legacy) |
| `dt` | `tree.DecisionTreeClassifier` | InputData; TensorData CPU (legacy) |
| `rf` | `ensemble.RandomForestClassifier` | InputData; TensorData CPU (legacy) |
| `mlp` | `neural_network.MLPClassifier` | InputData; TensorData CPU (legacy) |

### 2.5. Legacy sklearn regression

Стратегия: [SklearnRegressionStrategy](../../fedot/core/operations/strategies/models/sklearn.py).
Общий fit и mapping: [SklearnModelStrategy](../../fedot/core/operations/strategies/models/sklearn.py).

JSON допускает regression/TS. Native TensorData/CUDA adapter отсутствует; NumPy результат прикрепляется через общий output converter.

| ID | Estimator из sklearn | Формат |
| --- | --- | --- |
| `adareg` | `ensemble.AdaBoostRegressor` | InputData; TensorData CPU (legacy) |
| `gbr` | `ensemble.GradientBoostingRegressor` | InputData; TensorData CPU (legacy) |
| `dtreg` | `tree.DecisionTreeRegressor` | InputData; TensorData CPU (legacy) |
| `treg` | `ensemble.ExtraTreesRegressor` | InputData; TensorData CPU (legacy) |
| `rfr` | `ensemble.RandomForestRegressor` | InputData; TensorData CPU (legacy) |
| `linear` | `linear_model.LinearRegression` | InputData; TensorData CPU (legacy) |
| `ridge` | `linear_model.Ridge` | InputData; TensorData CPU (legacy) |
| `lasso` | `linear_model.Lasso` | InputData; TensorData CPU (legacy) |
| `svr` | `svm.LinearSVR` (не SVR) | InputData; TensorData CPU (legacy) |
| `sgdr` | `linear_model.SGDRegressor` | InputData; TensorData CPU (legacy) |

### 2.6. Legacy FEDOT classification wrappers

Стратегия: [AdapterClassificationStrategy](../../fedot/core/operations/strategies/models/adapters.py).

Для четырёх tabular wrappers выполнен CPU TensorData smoke-check. Они отдают NumPy, не сохраняют CUDA runtime.
`cnn` — другой контракт: image InputData, Keras/TensorFlow, `num_classes` и NumPy-операции.

| ID | Реализация | Где реализация | Формат |
| --- | --- | --- | --- |
| `lda` | `LDAImplementation` | [estimator_adapters.py](../../fedot/core/operations/implementations/models/estimator_adapters.py) | InputData; TensorData CPU (legacy) |
| `qda` | `QDAImplementation` | [estimator_adapters.py](../../fedot/core/operations/implementations/models/estimator_adapters.py) | InputData; TensorData CPU (legacy) |
| `svc` | `FedotSVCImplementation` (OneVsRestClassifier + SVC) | [estimator_adapters.py](../../fedot/core/operations/implementations/models/estimator_adapters.py) | InputData; TensorData CPU (legacy) |
| `knn` | `KnnClassificationAdapter` | [estimator_adapters.py](../../fedot/core/operations/implementations/models/estimator_adapters.py) | InputData; TensorData CPU (legacy) |
| `cnn` | `FedotCNNImplementation` | [keras.py](../../fedot/core/operations/implementations/models/keras.py) | InputData |

### 2.7. Legacy KNN regression и clustering

Стратегии: [AdapterRegressionStrategy](../../fedot/core/operations/strategies/models/adapters.py) и [SklearnClusteringStrategy](../../fedot/core/operations/strategies/models/sklearn.py).

KNN wrapper передаёт features/target в sklearn; clustering передаёт features напрямую. Native TensorData/CUDA-контракта нет.

| ID | Стратегия | Реализация и место | Формат |
| --- | --- | --- | --- |
| `knnreg` | `AdapterRegressionStrategy` | [estimator_adapters.py](../../fedot/core/operations/implementations/models/estimator_adapters.py) | InputData; TensorData CPU (legacy) |
| `kmeans` | `SklearnClusteringStrategy` | `sklearn.cluster.KMeans`, mapping в `strategies/models/sklearn.py` | InputData; TensorData CPU (legacy) |

### 2.8. Boosting

Стратегии: [boosting.py](../../fedot/core/operations/strategies/models/boosting.py).
Все FEDOT wrappers: [boosting.py](../../fedot/core/operations/implementations/models/boosting.py).

InputData-oriented. XGBoost/LightGBM используют DataFrame adapter, legacy split и
`get_not_encoded_data()`; в adapter есть `target.size` как NumPy property.
CatBoost использует Pool и legacy split/schema. GPU-возможности самих библиотек
не являются TensorData-поддержкой этих FEDOT wrappers.

| ID | Стратегия | Реализация | Формат |
| --- | --- | --- | --- |
| `xgboost` | `BoostingClassificationStrategy` | `FedotXGBoostClassificationImplementation` | InputData |
| `xgboostreg` | `BoostingRegressionStrategy` | `FedotXGBoostRegressionImplementation` | InputData |
| `lgbm` | `BoostingClassificationStrategy` | `FedotLightGBMClassificationImplementation` | InputData |
| `lgbmreg` | `BoostingRegressionStrategy` | `FedotLightGBMRegressionImplementation` | InputData |
| `catboost` | `BoostingClassificationStrategy` | `FedotCatBoostClassificationImplementation` | InputData |
| `catboostreg` | `BoostingRegressionStrategy` | `FedotCatBoostRegressionImplementation` | InputData |

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

BaseFoundationalImplementation возвращает OutputData через legacy implementation converter.
Он требует `encoded_idx`, `categorical_features`, `supplementary_data`, которых нет у
TensorData. TabPFN categorical path дополнительно вызывает `get_not_encoded_data()`.
Тот факт, что сторонняя библиотека может принимать tensors, не закрывает этот контракт.

| ID | Стратегия | Реализация | Формат |
| --- | --- | --- | --- |
| `tabpfn` | `TabPFNClassificationStrategy` | `FedotTabPFNClassificationImplementation` | InputData |
| `tabpfnreg` | `TabPFNRegressionStrategy` | `FedotTabPFNRegressionImplementation` | InputData |
| `tabicl` | `TabICLClassificationStrategy` | `FedotTabICLClassificationImplementation` | InputData |
| `tabiclreg` | `TabICLRegressionStrategy` | `FedotTabICLRegressionImplementation` | InputData |

### 2.11. Legacy custom model

Стратегия: [CustomModelStrategy](../../fedot/core/operations/strategies/models/custom.py).
Реализация: [CustomModelImplementation](../../fedot/core/operations/implementations/models/custom.py).

Callbacks получают idx/features/target/params, но wrapper возвращает OutputData
через legacy converter. Произвольный callback не делает wrapper TensorData-native.
Современные manifest extensions — другой механизм, см. раздел 6.

| ID | Реализация | Формат |
| --- | --- | --- |
| `custom` | `CustomModelImplementation` | InputData |

## 3. GPU-каталог — gpu_models_repository.json

Здесь ID вроде `logit`/`rf` означают другие реализации, чем в основном каталоге.
`preset='gpu'` выбирает repository/strategy; он не переносит любую обычную модель на GPU.

Общая cuML база: [cuml.py](../../fedot/core/operations/strategies/models/cuml.py).
Адаптация параметров, выбор dtype, проверка окружения и запрет solver aliases находятся в общей стратегии.
Mapping estimator-классов расположен в `strategies/models/cuml.py`, а не в отдельных model wrappers.

После последнего перевода cuML принимает только TensorData. CUDA Torch → CuPy → cuML
и CuPy → Torch используют DLPack. CPU TensorData копируется на CUDA для обучения;
предсказание возвращается на исходный CPU. Требуются native Linux, RAPIDS и CUDA;
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
| `sgd` | `cuml.solvers.SGD` — low-level solver; fit явно отклоняется | Нет рабочего fit-контракта |

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

### 3.4. Foundational GPU aliases

Те же [foundational strategies](../../fedot/core/operations/strategies/models/foundation.py) и [foundational implementations](../../fedot/core/operations/implementations/models/foundation.py), что в разделе 2.10.

GPU alias не меняет контейнерный контракт. Эти четыре строки не являются cuML-моделями.

| ID | Стратегия | Реализация | Формат |
| --- | --- | --- | --- |
| `tabpfn_gpu` | `TabPFNClassificationStrategy` | `FedotTabPFNClassificationImplementation` | InputData |
| `tabpfnreg_gpu` | `TabPFNRegressionStrategy` | `FedotTabPFNRegressionImplementation` | InputData |
| `tabicl_gpu` | `TabICLClassificationStrategy` | `FedotTabICLClassificationImplementation` | InputData |
| `tabiclreg_gpu` | `TabICLRegressionStrategy` | `FedotTabICLRegressionImplementation` | InputData |

## 4. Data operations — data_operation_repository.json

Не путать legacy node ID `scaling`, `simple_imputation`, `one_hot_encoding` с
одноимёнными методами новых preprocessing services. Services могут поддерживать
TensorData, пока старые pipeline nodes всё ещё ждут InputData.

### 4.1. TensorData-native transforms и optional preprocessing

Стратегии: [ProjectionDataOperationStrategy](../../fedot/core/operations/strategies/data_operations/projection.py) и [OptionalPreprocessingStrategy](../../fedot/core/operations/strategies/data_operations/optional.py).

PCA/SVD сохраняют Torch runtime и возвращают transformed `features`, а не model `predict`.
Для optional preprocessing в core зарегистрирован **только tabular runtime**:
[tensor_optional_runtime.py](../../fedot/preprocessing/service/tensor_optional_runtime.py).
TS/text/image runtime должен быть зарегистрирован extension; широкие JSON tasks/input_types
не доказывают наличие такого runtime в core.

| ID | Стратегия | Реализация и файл | Формат |
| --- | --- | --- | --- |
| `pca` | `ProjectionDataOperationStrategy` | [features_reducing.py](../../fedot/core/operations/implementations/data_operations/features_reducing.py) | TensorData |
| `truncated_svd` | `ProjectionDataOperationStrategy` | [features_reducing.py](../../fedot/core/operations/implementations/data_operations/features_reducing.py) | TensorData |
| `optional_preprocessing` | `OptionalPreprocessingStrategy` | [OptionalService](../../fedot/preprocessing/service/optional_service.py) / [OptionalTabularService](../../fedot/preprocessing/service/tabular_optional_service.py) | TensorData |

### 4.2. Legacy общие tabular transforms

Стратегия для всех строк: [PreprocessingDataOperationStrategy](../../fedot/core/operations/strategies/data_operations/common_preprocessing.py).

Scaling/normalization/poly_features наследуют EncodedInvariantImplementation с
NumPy/Pandas branching и `.iloc`. Остальные legacy transforms используют OutputData,
`encoded_idx` и column metadata. TensorData-поддержку новых services здесь не учитывать.

| ID | Реализация | Файл | Формат |
| --- | --- | --- | --- |
| `scaling` | `ScalingImplementation` (StandardScaler) | [sklearn_transformations.py](../../fedot/core/operations/implementations/data_operations/sklearn_transformations.py) | InputData |
| `normalization` | `NormalizationImplementation` (MinMaxScaler) | [sklearn_transformations.py](../../fedot/core/operations/implementations/data_operations/sklearn_transformations.py) | InputData |
| `simple_imputation` | `ImputationImplementation` | [sklearn_transformations.py](../../fedot/core/operations/implementations/data_operations/sklearn_transformations.py) | InputData |
| `dask_pca` | `DaskPCAImplementation` | [features_reducing.py](../../fedot/core/operations/implementations/data_operations/features_reducing.py) | InputData |
| `kernel_pca` | `KernelPCAImplementation` | [features_reducing.py](../../fedot/core/operations/implementations/data_operations/features_reducing.py) | InputData |
| `fast_ica` | `FastICAImplementation` | [features_reducing.py](../../fedot/core/operations/implementations/data_operations/features_reducing.py) | InputData |
| `poly_features` | `PolyFeaturesImplementation` | [sklearn_transformations.py](../../fedot/core/operations/implementations/data_operations/sklearn_transformations.py) | InputData |
| `one_hot_encoding` | `OneHotEncodingImplementation` | [categorical_encoders.py](../../fedot/core/operations/implementations/data_operations/categorical_encoders.py) | InputData |
| `label_encoding` | `LabelEncodingImplementation` | [categorical_encoders.py](../../fedot/core/operations/implementations/data_operations/categorical_encoders.py) | InputData |
| `topological_features` | `TopologicalFeaturesImplementation` | [fast_topological_extractor.py](../../fedot/core/operations/implementations/data_operations/topological/fast_topological_extractor.py) | InputData |

### 4.3. Legacy regression preprocessing

Стратегия для всех строк: [RegressionDataOperationStrategy](../../fedot/core/operations/strategies/data_operations/preprocessing.py).

Selectors/filters возвращают OutputData через legacy converter. Decompose дополнительно использует supplementary_data.compound_mask/flow_mask для разделения входов родителей.

| ID | Реализация | Файл | Формат |
| --- | --- | --- | --- |
| `ransac_lin_reg` | `LinearRegRANSACImplementation` | [sklearn_filters.py](../../fedot/core/operations/implementations/data_operations/sklearn_filters.py) | InputData |
| `ransac_non_lin_reg` | `NonLinearRegRANSACImplementation` | [sklearn_filters.py](../../fedot/core/operations/implementations/data_operations/sklearn_filters.py) | InputData |
| `isolation_forest_reg` | `IsolationForestRegImplementation` | [sklearn_filters.py](../../fedot/core/operations/implementations/data_operations/sklearn_filters.py) | InputData |
| `rfe_lin_reg` | `LinearRegFSImplementation` | [sklearn_selectors.py](../../fedot/core/operations/implementations/data_operations/sklearn_selectors.py) | InputData |
| `rfe_non_lin_reg` | `NonLinearRegFSImplementation` | [sklearn_selectors.py](../../fedot/core/operations/implementations/data_operations/sklearn_selectors.py) | InputData |
| `decompose` | `DecomposerRegImplementation` | [decompose.py](../../fedot/core/operations/implementations/data_operations/decompose.py) | InputData |

### 4.4. Legacy classification preprocessing

Стратегия для всех строк: [ClassificationDataOperationStrategy](../../fedot/core/operations/strategies/data_operations/preprocessing.py).

`resample` требует `encoded_idx`, меняет строки/target и использует NumPy/Pandas.
На TensorData CPU фактически воспроизведён AttributeError по `encoded_idx`.
Его hard-coded insertion в mutation остаётся blocker поиска. `class_decompose`
зависит от legacy metadata входов родителей.

| ID | Реализация | Файл | Формат |
| --- | --- | --- | --- |
| `rfe_lin_class` | `LinearClassFSImplementation` | [sklearn_selectors.py](../../fedot/core/operations/implementations/data_operations/sklearn_selectors.py) | InputData |
| `rfe_non_lin_class` | `NonLinearClassFSImplementation` | [sklearn_selectors.py](../../fedot/core/operations/implementations/data_operations/sklearn_selectors.py) | InputData |
| `isolation_forest_class` | `IsolationForestClassImplementation` | [sklearn_filters.py](../../fedot/core/operations/implementations/data_operations/sklearn_filters.py) | InputData |
| `class_decompose` | `DecomposerClassImplementation` | [decompose.py](../../fedot/core/operations/implementations/data_operations/decompose.py) | InputData |
| `resample` | `ResampleImplementation` | [resampling.py](../../fedot/core/operations/implementations/data_operations/resampling.py) | InputData |

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

Fit возвращает `object()`; predict создаёт OutputData с исходными features.
TensorData shell может переупаковать этот passthrough в TensorData без NumPy copy.
На CPU проверен `data_source_table`. Остальные строки используют тот же код,
но raw text/image/TS и multimodal API не проходили отдельный acceptance test.

| ID | Назначение | Формат |
| --- | --- | --- |
| `data_source_table` | Tabular source | InputData; TensorData через OutputData |
| `data_source_ts` | TS source | InputData; TensorData через OutputData |
| `data_source_img` | Image source | InputData; TensorData через OutputData |
| `data_source_text` | Text source | InputData; TensorData через OutputData |

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

| Каталог | TensorData-контракт | Остальное |
| --- | ---: | --- |
| Основные модели | 18 | 22 legacy-модели с возможной CPU TensorData-совместимостью; 22 InputData-oriented |
| GPU модели | 16 cuML | 4 foundational InputData aliases; 1 отвергаемый solver alias |
| Data operations | 3 | 32 InputData-oriented; 4 passthrough data sources через OutputData |
| AutoML | 0 | 4 InputData-oriented |

18 основных TensorData моделей — 4 базовых Torch, 6 sklearn/InterpretML adapters,
8 tabular neural. Частичная CPU-совместимость не должна использоваться как
TensorData-compatible allowlist для evolutionary search.

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

1. Решить судьбу оставшихся legacy sklearn/tabular wrappers: явный TensorData adapter
   или замена; устранить NumPy predictions внутри TensorData там, где нужен Torch runtime.
2. Перенести или исключить из compatible pool legacy tabular data operations.
   В первую очередь `resample`, hard-coded mutations и initial assumptions:
   несовместимый узел не должен попадать в TensorData graph.
3. Отдельно перенести boosting, foundational и legacy custom wrappers.
   GPU training сторонней библиотеки и TensorData-контракт — отдельные задачи.
4. TS models/transforms, текст, изображения и multimodal — отдельные acceptance gates;
   не выводить их готовность из того, что общий pipeline принимает TensorData.
5. CV/OOF, sampling/chunks, remote evaluation и auxiliary API лежат вне этой карты
   operations. Даже полностью migrated allowlist сам по себе не закрывает полный search.

Подробный контекст search и последовательность этапов:
[fedot_tensordata_handoff_2026_10_08.md](fedot_tensordata_handoff_2026_10_08.md)
и [tensordata_full_pipeline_search_plan.md](tensordata_full_pipeline_search_plan.md).

## 8. Как проверялось и как обновлять

Основа аудита — JSON metadata → конкретная strategy → estimator/implementation →
output converter. Сверены все 126 записей четырёх каталогов и operation mappings
стратегий. Формат не определялся только по type hints.

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
