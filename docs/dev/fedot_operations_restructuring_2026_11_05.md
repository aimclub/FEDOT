# Результат реструктуризации операций FEDOT

Дата: 6 ноября 2026 года. Ветка `fedot_1.0.0_new`.
Подготовительные изменения моделей сохранены отдельно в `cd281e48`.

## Что сделано

- Стратегии находятся в `fedot/core/operations/strategies/`, реализации — в соседнем `implementations/`.
- В обоих пакетах модели отделены от data operations. Classification/regression одного семейства остаются вместе.
- `EvaluationStrategy`, `Operation`, `Model`, `DataOperation` и operation IDs сохраняют имена и назначение.
- Общие implementation-контракты собраны в `implementations/base.py`; локальные обучающие базы остаются с семейством.
- Учтён весь состав прежних модулей, включая selectors, filters, decomposition, text, forecasting и helpers.
- JSON-каталоги, внутренние импорты, тесты и Sphinx API docs используют новые пути.
- Старый пакет `evaluation/` удалён целиком; переэкспортов и дополнительных импортных aliases нет.

## Согласованное объединение нейросетевых стратегий

`Simple*Strategy` и `TensorTabular*Strategy` объединены в `NeuralTabular*Strategy`.
У Linear/MLP и TabM/RTDL/RealMLP теперь общий контракт реализации:

```text
fit(TensorData) -> self
classifier.predict(TensorData) -> TensorData с метками исходных классов
classifier.predict_proba(TensorData) -> TensorData с матрицей [n_samples, n_classes]
regressor.predict(TensorData) -> TensorData с предсказанием
```

Бинарное сокращение вероятностей происходит в общей стратегии. Режимы `labels`,
`probs`, `full_probs`, `default` и `False` поддерживаются. Метаданные и исходный
device сохраняются, входной контейнер не изменяется.
Обучающие циклы, архитектуры, scaling и алгоритмы реализации не объединялись.
Seed Torch-моделей инициализируется до общего random-state context, чтобы после fit восстановить его значение.

## Решения и границы

- `TensorSkLearn*` пока сохраняют имена и отдельный контракт в `strategies/models/sklearn.py` рядом с legacy sklearn.
  После миграции остальных моделей нужно объединить классы и убрать `Tensor` из имени; повторный перенос файла не нужен.
- `DataOperationImplementation` и `TensorDataOperationImplementation` сосуществуют в `implementations/base.py` до согласования контрактов.
- KernelPCA/FastICA/DaskPCA и их component-analysis база перенесены к PCA/SVD в `features_reducing.py`.
  Их InputData-контракт и прежний preprocessing dispatch сохранены; общий projection dispatch появится при TensorData-адаптации.
- Dask imports стали локальными: импорт Torch PCA/SVD не требует DaskML.
- EBM/KNN/SVC/LDA/QDA adapters сгруппированы в `models/estimator_adapters.py`.
  Прямые внешние estimators остались ссылками в strategy mapping.
- Старые обученные пайплайны следует запускать на версии FEDOT и зависимостей, на которых они обучались.
  Поддержка и конверсия старого сохранённого состояния отложены; загрузчики и формат сохранения не расширялись.
- За пределами `core/operations` обновлены ссылки на классы: каталоги, pipeline import/load helpers, extensions,
  random-state helper, встроенный Industrial, тесты и документация. Логика API, merger, cache, поиска и метрик не менялась.

## Пересечения с Industrial

См. [заметки о синхронизации](fedot_industrial_tensordata_sync_2026_11_05.md).
Самостоятельный локальный репозиторий Industrial не изменялся; потребители обновят импорты при синхронизации веток.

Перед native pipeline search / CV / mergers / cache:

1. Сверить integration patches Industrial с новым кодом FEDOT по смыслу.
2. Проверить согласованность `idx` и строк при feature fusion, identity/fingerprint кеша предсказаний.
3. Завершить TensorData-native CV/OOF и task-specific output contracts.

## Проверки

- `tests/core/operations`: **183 passed, 25 skipped** (CUDA недоступна).
  Проверены neural families, binary/multiclass, output modes, metadata и обычный `Pipeline.save/load` для новых пайплайнов.
- При расширенной проверке переноса прошло 1288 тестов; 30 несетевых падений повторены на `cd281e48`,
  ещё одно связано с загрузкой датасета. Полный набор блокируют 37 прежних ошибок сбора.
  Эти проблемы не исправлялись в структурном изменении.
- Sphinx dummy build выполнен без предупреждений с отключённым внешним intersphinx inventory.
- AST-аудит подтвердил наличие всех перенесённых собственных классов/функций.

## Полная карта модулей

Все пути ниже относительно `fedot/core/operations/`. Старые пути приведены для истории и удалены из дерева.
При разделении исходного модуля указаны все места, куда переехали его собственные классы и функции.

| Раньше | Каноническое расположение |
| --- | --- |
| `evaluation/abstract_node.py` | `implementations/base.py` |
| `evaluation/automl.py` | `strategies/automl.py` |
| `evaluation/boostings.py` | `strategies/models/boosting.py` |
| `evaluation/classification.py` | `strategies/models/sklearn.py`<br>`strategies/models/adapters.py`<br>`strategies/data_operations/preprocessing.py` |
| `evaluation/clustering.py` | `strategies/models/sklearn.py` |
| `evaluation/common_preprocessing.py` | `strategies/data_operations/common_preprocessing.py` |
| `evaluation/custom.py` | `strategies/models/custom.py` |
| `evaluation/data_source.py` | `strategies/sources.py` |
| `evaluation/evaluation_interfaces.py` | `strategies/base.py`<br>`strategies/models/sklearn.py` |
| `evaluation/extensions.py` | `strategies/extensions.py` |
| `evaluation/foundational.py` | `strategies/models/foundation.py` |
| `evaluation/gpu/classification.py` | `strategies/models/cuml.py` |
| `evaluation/gpu/clustering.py` | `strategies/models/cuml.py` |
| `evaluation/gpu/common.py` | `strategies/models/cuml.py` |
| `evaluation/gpu/regression.py` | `strategies/models/cuml.py` |
| `evaluation/operation_implementations/data_operations/categorical_encoders.py` | `implementations/data_operations/categorical_encoders.py` |
| `evaluation/operation_implementations/data_operations/decompose.py` | `implementations/data_operations/decompose.py` |
| `evaluation/operation_implementations/data_operations/features_reducing.py` | `implementations/data_operations/features_reducing.py` |
| `evaluation/operation_implementations/data_operations/sklearn_filters.py` | `implementations/data_operations/sklearn_filters.py` |
| `evaluation/operation_implementations/data_operations/sklearn_imbalanced_class.py` | `implementations/data_operations/resampling.py` |
| `evaluation/operation_implementations/data_operations/sklearn_selectors.py` | `implementations/data_operations/sklearn_selectors.py` |
| `evaluation/operation_implementations/data_operations/sklearn_transformations.py` | `implementations/data_operations/sklearn_transformations.py`<br>`implementations/data_operations/features_reducing.py` |
| `evaluation/operation_implementations/data_operations/text_preprocessing.py` | `implementations/data_operations/text_preprocessing.py` |
| `evaluation/operation_implementations/data_operations/text_pretrained.py` | `implementations/data_operations/text_pretrained.py` |
| `evaluation/operation_implementations/data_operations/topological/fast_topological_extractor.py` | `implementations/data_operations/topological/fast_topological_extractor.py` |
| `evaluation/operation_implementations/data_operations/topological/topological_backend.py` | `implementations/data_operations/topological/topological_backend.py` |
| `evaluation/operation_implementations/data_operations/topological/topological_backend_rules.py` | `implementations/data_operations/topological/topological_backend_rules.py` |
| `evaluation/operation_implementations/data_operations/ts_transformations.py` | `implementations/data_operations/ts_transformations.py` |
| `evaluation/operation_implementations/implementation_interfaces.py` | `implementations/base.py`<br>`implementations/data_operations/sklearn_transformations.py` |
| `evaluation/operation_implementations/models/boostings_implementations.py` | `implementations/models/boosting.py` |
| `evaluation/operation_implementations/models/custom_model.py` | `implementations/models/custom.py` |
| `evaluation/operation_implementations/models/discriminant_analysis.py` | `implementations/models/estimator_adapters.py` |
| `evaluation/operation_implementations/models/foundational.py` | `implementations/models/foundation.py` |
| `evaluation/operation_implementations/models/keras.py` | `implementations/models/keras.py` |
| `evaluation/operation_implementations/models/knn.py` | `implementations/models/estimator_adapters.py` |
| `evaluation/operation_implementations/models/svc.py` | `implementations/models/estimator_adapters.py` |
| `evaluation/operation_implementations/models/tensor_tabular.py` | `implementations/models/neural.py` |
| `evaluation/operation_implementations/models/torch.py` | `implementations/models/neural.py` |
| `evaluation/operation_implementations/models/ts_implementations/arima.py` | `implementations/models/forecasting/arima.py` |
| `evaluation/operation_implementations/models/ts_implementations/cgru.py` | `implementations/models/forecasting/cgru.py` |
| `evaluation/operation_implementations/models/ts_implementations/naive.py` | `implementations/models/forecasting/naive.py` |
| `evaluation/operation_implementations/models/ts_implementations/poly.py` | `implementations/models/forecasting/poly.py` |
| `evaluation/operation_implementations/models/ts_implementations/statsmodels.py` | `implementations/models/forecasting/statsmodels.py` |
| `evaluation/operation_implementations/rules.py` | `implementations/data_operations/rules.py` |
| `evaluation/operation_implementations/schema.py` | `implementations/data_operations/schema.py` |
| `evaluation/operation_implementations/tools.py` | `implementations/data_operations/tools.py` |
| `evaluation/optional_preprocessing.py` | `strategies/data_operations/optional.py` |
| `evaluation/optional_preprocessing_strategy_builder.py` | `strategies/data_operations/optional_config.py` |
| `evaluation/regression.py` | `strategies/models/sklearn.py`<br>`strategies/models/adapters.py`<br>`strategies/data_operations/preprocessing.py` |
| `evaluation/simple_classification.py` | `strategies/models/neural.py` |
| `evaluation/simple_regression.py` | `strategies/models/neural.py` |
| `evaluation/tensor_models.py` | `strategies/models/neural.py` |
| `evaluation/tensor_sklearn.py` | `strategies/models/sklearn.py`<br>`implementations/models/estimator_adapters.py` |
| `evaluation/tensor_transform.py` | `strategies/data_operations/projection.py` |
| `evaluation/text.py` | `strategies/data_operations/text.py` |
| `evaluation/time_series.py` | `strategies/models/forecasting.py`<br>`strategies/data_operations/time_series.py` |

## Переименованные классы

| Старое имя | Каноническое имя |
| --- | --- |
| `CumlClusteringStrategy` | `CuMLClusteringStrategy` |
| `CuMLEvaluationStrategy` | `CuMLModelStrategy` |
| `FedotClassificationPreprocessingStrategy` | `ClassificationDataOperationStrategy` |
| `FedotClassificationStrategy` | `AdapterClassificationStrategy` |
| `FedotFTTransformerClassificationImplementation` | `FTTransformerClassifier` |
| `FedotFTTransformerImplementation` | `FTTransformerImplementation` |
| `FedotFTTransformerRegressionImplementation` | `FTTransformerRegressor` |
| `FedotKnnClassImplementation` | `KnnClassificationAdapter` |
| `FedotKnnRegImplementation` | `KnnRegressionAdapter` |
| `FedotPreprocessingStrategy` | `PreprocessingDataOperationStrategy` |
| `FedotRealMLPClassificationImplementation` | `RealMLPClassifier` |
| `FedotRealMLPImplementation` | `RealMLPImplementation` |
| `FedotRealMLPRegressionImplementation` | `RealMLPRegressor` |
| `FedotRegressionPreprocessingStrategy` | `RegressionDataOperationStrategy` |
| `FedotRegressionStrategy` | `AdapterRegressionStrategy` |
| `FedotResNetClassificationImplementation` | `ResNetClassifier` |
| `FedotResNetImplementation` | `ResNetImplementation` |
| `FedotResNetRegressionImplementation` | `ResNetRegressor` |
| `FedotRTDLImplementation` | `RTDLImplementation` |
| `FedotTabMClassificationImplementation` | `TabMClassifier` |
| `FedotTabMImplementation` | `TabMImplementation` |
| `FedotTabMRegressionImplementation` | `TabMRegressor` |
| `FedotTextPreprocessingStrategy` | `TextCleaningStrategy` |
| `FedotTsForecastingStrategy` | `ForecastingStrategy` |
| `FedotTsTransformingStrategy` | `TimeSeriesDataOperationStrategy` |
| `GensimTextVectorizeStrategy` | `PretrainedEmbeddingStrategy` |
| `SimpleClassificationStrategy` | `NeuralTabularClassificationStrategy` |
| `SimpleRegressionStrategy` | `NeuralTabularRegressionStrategy` |
| `SkLearnClassificationStrategy` | `SklearnClassificationStrategy` |
| `SkLearnClusteringStrategy` | `SklearnClusteringStrategy` |
| `SkLearnEvaluationStrategy` | `SklearnModelStrategy` |
| `SkLearnRegressionStrategy` | `SklearnRegressionStrategy` |
| `SkLearnTextVectorizeStrategy` | `SklearnTextVectorizationStrategy` |
| `TensorOptionalPreprocessingStrategy` | `OptionalPreprocessingStrategy` |
| `TensorTabularClassificationImplementation` | `NeuralTabularClassificationImplementation` |
| `TensorTabularClassificationStrategy` | `NeuralTabularClassificationStrategy` |
| `TensorTabularModelImplementation` | `NeuralTabularModelImplementation` |
| `TensorTabularRegressionStrategy` | `NeuralTabularRegressionStrategy` |
| `TensorTabularStrategy` | `NeuralTabularStrategy` |
| `TensorTransformStrategy` | `ProjectionDataOperationStrategy` |

Остальные классы сохранили имена. Карту текущих форматов см. в [отдельном аудите](fedot_operations_data_formats_2026_10_09.md).
