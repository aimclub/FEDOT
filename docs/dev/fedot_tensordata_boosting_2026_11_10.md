# Перенос бустингов на TensorData

10 ноября 2026. Завершён пункт 1 [порядка переноса](fedot_tensordata_operations_migration_order_2026_11_07.md).

## Что изменено

- `fedot/core/operations/implementations/models/boosting.py`: общий `BoostingImplementation`;
  TensorData fit/predict для CatBoost, XGBoost и LightGBM, без InputData/DataFrame adapter.
  Алгоритмы остаются библиотечными. Общая база содержит только реально повторявшуюся подготовку.
- `fedot/core/operations/strategies/models/boosting.py`: обе стратегии принимают/возвращают TensorData;
  реализации возвращают Torch, стратегия упаковывает predictions с сохранением idx/target/метаданных.
- `fedot/core/operations/implementations/base.py`: hook `prepare_categorical_features(features, *, fit=False)`;
  default ничего не меняет. Флаг позволяет адаптеру обучить словарь только на fit.
- `fedot/core/data/split/data_split.py`: необязательная стратификация tensor hold-out с прежним seed 42;
  numerical features остаются на устройстве, sklearn видит только labels и позиции строк.
  Без нового флага прежний последовательный split сохранён. Это необходимая правка вне operations.
- `tests/core/operations/test_boosting_contract.py`: TensorData contracts, stopping parameters, multioutput,
  устойчивый словарь категорий, imputation/no-imputation, predict без target и композиция узлов.
- `tests/core/data/test_tensor_data_split.py`: пропорции split и выравнивание row fields.

В рамках этого продолжения default params и alias rules повторно не менялись: согласованные ранее три
исправления уже лежали в рабочем дереве. Другие прежние изменения категорий/preprocessing также сохранены.
Обновлены карта форматов, чек-лист переноса и список вопросов для экспериментов.

## Данные и устройства

| Семейство | Граница библиотеки | Явный GPU-параметр |
| --- | --- | --- |
| XGBoost | Torch напрямую, CPU/CUDA; библиотека читает данные через NumPy / CUDA Array Interface | `device='cuda:0'`, `tree_method='hist'` |
| CatBoost | Pool/FeaturesData: числовой float32-блок и строковый блок категорий в памяти CPU | `task_type='GPU'`, `devices='0'` |
| LightGBM | NumPy в памяти CPU, `categorical_feature` задаёт категориальные позиции | `device_type='gpu'` (проверен OpenCL) |

Подготовка числового блока, объединение блоков и упаковка predictions выполняются Torch.
XGBoost 3.0.5 принимает Torch tensors напрямую, включая features/target/eval и predict.
Явный NumPy/CuPy adapter удалён; остаётся только detach/перенос на устройство модели/contiguous.
Уже подходящий tensor не копируется. На CPU библиотека внутри использует NumPy view,
на CUDA — CUDA Array Interface. Исходные строковые категории по-прежнему требуют библиотечной подготовки.
CatBoost/LightGBM через используемые Python API требуют host input даже для GPU-обучения.
CUDA predict XGBoost теперь обходит NumPy-выход sklearn: Booster.inplace_predict → CuPy → Torch DLPack.
Сохранены best iteration, labels/probabilities, multiclass/multilabel и multi:softmax.
CPU/gblinear продолжают использовать sklearn. Итоговое устройство соответствует входному TensorData.
Автоматического выбора устройства/новых GPU-ID нет; GPU задаётся библиотечными параметрами.
Прямой импорт CuPy из FEDOT boosting удалён; установленный XGBoost использует CuPy внутри CUDA-пути.
Обновление XGBoost не потребовалось.

При `enable_categorical=False` используется текущая features-матрица. При True восстанавливаются активные
исходные категории через TensorData.get_not_encoded_data(); числовые преобразования не отменяются.
XGBoost/LightGBM используют существующий LabelEncoder с vocabulary, обученным на train; unknown → NaN.
CatBoost использует строки; исходный нормализованный NaN представлен строкой `nan`.
Адаптеры не выполняют статистическую импутацию. Optional imputation заранее согласованно изменяет оба
представления; её отсутствие сохраняет missing-value semantics.

## Сохранённая логика

Внутренний eval split 80/20, seed=42, политика стратификации, eval metrics и multioutput без eval сохранены.
Стратификация позже перенесена на Torch: конкретные случайные строки отличаются от sklearn,
повторяемость сохраняется на одном устройстве. См. [native rows](fedot_tensor_native_rows_2026_11_10.md).
Default patience и выбор best iteration не менялись. Удалена пустая обёртка callbacks LightGBM:
early stopping по-прежнему создаётся самой библиотекой из параметра.
Как и раньше, XGBoost/LightGBM не получают FEDOT n_jobs; CatBoost получает его как thread_count.
Общую политику ресурсов нужно согласовать отдельно, а не менять скрыто при переносе контейнера.
Исходные параметры стратегии копируются; single/multi-target fit не меняет общий класс реализации.

Сопутствующие небольшие исправления: XGBoost feature_importances_ вместо опечатки;
feature importance plot больше не требует pandas; CatBoost.load_model сохраняет тип classifier/regressor.

До миграции CatBoost.fit уже получал Pool из InputData.features/target и categorical_idx;
FeaturesData не использовался. Теперь числовой Torch-блок конвертируется в CPU NumPy float32,
исходный категориальный NumPy-блок приводится к строкам/object; они передаются отдельно в FeaturesData → Pool.
Это позволяет не собирать общую object-матрицу из числовых и строковых колонок.

## Проверки и пределы

- 21 детерминированный короткий fit до/после: binary/multiclass/regression, eval/no eval, multioutput;
  все 45 массивов predictions точно совпали с InputData-baseline (после трёх согласованных исправлений).
- 80 адресных CPU-тестов: boosting, параметры, split, сохранение категорий и fit-stage contracts.
- 9 коротких GPU fits на RTX 4080 SUPER: classification, regression, категории с NaN для каждого семейства;
  eval set включён, predictions конечные, Torch и на устройстве входа. XGBoost CUDA и CatBoost GPU
  подтверждены native config, LightGBM OpenCL GPU — параметрами/выводом библиотеки.
- Pipeline.fit и root_node.predict для optional_preprocessing → каждого бустинга проверены.
  Полный поиск/API/CV и нагрузочные эксперименты — следующие задачи.
- После удаления входного адаптера XGBoost: 12 адресных CPU-тестов и 3 GPU fits;
  15 массивов XGBoost точно совпали с прежним InputData-baseline.
  Отдельно проверен Booster.inplace_predict на CUDA: он возвращает CuPy, значения совпали с текущими
  regression/binary probabilities при явном iteration_range до best_iteration + 1.
  Этот первоначальный эксперимент позже заменён основной CUDA-реализацией с сохранением классов,
  objective, multilabel и выбора итераций. См. следующий отчёт native rows.

Временные воспроизводящие скрипты и результаты: `/tmp/fedot_boosting_tensor_baseline_2026_11_10.py`,
`/tmp/fedot_boosting_tensor_{before,after}.{npz,json}`,
`/tmp/fedot_boosting_tensor_gpu_smoke_2026_11_10.py`,
`/tmp/fedot_boosting_tensor_gpu_results_2026_11_10.json`.
Они не являются постоянным хранилищем результатов.

Остаются отдельными: ragged multiclass multioutput, качество/стоимость eval split и patience,
CPU/GPU preferences. GPU predict и tensor split уже реализованы; см.
[native rows](fedot_tensor_native_rows_2026_11_10.md) и [вопросы экспериментов](fedot_model_experiment_questions_2026_11_10.md).
В smoke композиции данные создаются через create_data: optional service пока требует fingerprint,
поэтому прямой TensorData без fingerprint выявил отдельную проблему runtime, не связанную с бустингами.
При доработке полного TensorData runtime нужно согласовать этот контракт, не маскировать проблему в модели.
