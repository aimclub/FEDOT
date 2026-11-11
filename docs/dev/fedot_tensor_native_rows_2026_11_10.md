# Torch-разбиение, строки и GPU output XGBoost

10 ноября 2026. Продолжение переноса бустингов: удалены преобразования ради sklearn/NumPy там,
где данные уже были числовыми Torch tensors. Новые CPU/GPU-каталоги и автоматический device policy не вводились.

## Основные изменения и файлы

| Файл | Изменение |
| --- | --- |
| `fedot/core/data/split/tensor_split.py` | 44 строки: общая стратификация индексов через Torch unique/randperm/argsort/scatter |
| `fedot/core/data/split/data_split.py` | TensorData split использует новый helper; последовательный default сохранён; удалены проверки уже гарантированного row alignment |
| `fedot/core/data/tensor_data/row_operations.py` | Общие take_rows, common_row_indices и row_positions для split/merger |
| `fedot/core/data/tensor_data/contracts.py` | Torch boolean masks применяются на устройстве; нет промежуточного NumPy |
| `fedot/core/data/tensor_data/tools.py` | Сравнение одинаково размещённых tensors не копирует их на CPU |
| `fedot/core/data/merge/data_merger.py` | Torch row labels пересекаются/выравниваются на устройстве; смешанные числовые host labels переносятся к ним |
| `fedot/core/operations/implementations/models/boosting.py` | CUDA predict/proba XGBoost через Booster.inplace_predict → CuPy → Torch DLPack |
| `tests/core/data/test_tensor_row_operations.py` | Инварианты стратификации, выбор/согласование строк, запрет cpu/numpy/tolist на tensor-пути; CPU/CUDA |
| `tests/core/data/test_tensor_data_split.py` | Пропорции вместо точного sklearn RNG, выравнивание sidecar и prediction-only split |
| `tests/core/operations/test_boosting_contract.py` | GPU parity с sklearn для regression/binary/multiclass/multilabel/softmax/DART |
| `examples/benchmark/tensor_stratified_split.py` | Сравнение старого host/sklearn split и Torch indices |
| `examples/benchmark/xgboost_cuda_prediction.py` | Теперь сравнивает native sklearn с уже включённой FEDOT GPU-реализацией |

Обновлены карта форматов, результат переноса бустингов, общие требования, экспериментальные вопросы
и записка для синхронизации Industrial. Ранее существовавшие изменения рабочего дерева сохранены.

## Почему небольшой собственный helper

У Torch random_split нет стратификации; он создаёт Subset с Python-списком индексов.
Новый helper использует только готовые Torch-примитивы, без новой зависимости и без NumPy/sklearn.
Train counts рассчитываются пропорционально классам методом наибольших остатков;
остатки и порядок строк рандомизируются локальным генератором на устройстве target.
Суммарные размеры точные, индекс каждой строки встречается ровно в одном subset.
Multioutput labels могут стратифицироваться по комбинациям строк; boosting multioutput по-прежнему не использует eval.
Для слишком редких strata сохранён последовательный fallback, как в прежней политике.
Не меняются default split_ratio=0.8, seed=42, eval metrics, patience и выбор best iteration.

Конкретные случайные строки теперь отличаются от sklearn при том же seed; CPU/CUDA RNG также различаются.
Воспроизводимость гарантируется на одном устройстве. Равенство старых 45 prediction arrays относилось
к этапу до этой замены RNG; после неё нельзя требовать тех же predictions для classification с eval split.
При неизменном fitted model GPU-предсказания сверены со sklearn отдельно.

## Измерения на RTX 4080 SUPER

Стратификация: 5 классов, CUDA labels, итоговые индексы на CUDA, 3 прогрева / 9 повторов,
синхронизация CUDA до/после каждого замера; ниже медианы.

| Строк | Прежний host/sklearn путь, мс | Torch, мс | Ускорение этапа |
| --- | ---: | ---: | ---: |
| 1 000 | 0,662 | 0,517 | 1,28× |
| 12 000 | 2,062 | 0,612 | 3,37× |
| 250 000 | 38,713 | 1,125 | 34,41× |
| 1 000 000 | 153,934 | 2,011 | 76,53× |

Это время построения индексов, не ускорение всего fit. Порядок случайных строк отличается, пропорции сохраняются.
Срезы features и сам алгоритм XGBoost в этот замер не входят.
Все повторы: [split JSON](fedot_tensor_split_benchmark_2026_11_10.json).

Повторный prediction benchmark уже через production implementation: для 250 000 строк regression
1,391 → 0,968 мс (1,44×), binary probabilities 1,515 → 0,961 мс (1,58×), binary labels
2,512 → 0,997 мс (2,52×), multiclass probabilities 8,478 → 7,280 мс (1,16×), labels
12,003 → 7,453 мс (1,61×). Все 15 сравниваемых результатов совпали точно.
[Prediction JSON](fedot_xgboost_native_runtime_prediction_2026_11_10.json).

## Что теперь остаётся на CPU

Для числового TensorData с features/target/idx на CUDA и без host sidecar наша модельная цепочка
split → fit XGBoost → predict больше не переносит эти массивы в CPU NumPy.
Обучение по-прежнему выполняет XGBoost; CuPy для GPU output используется внутри библиотеки,
а Torch получает её буфер через DLPack. CPU predict сохраняет NumPy → Torch без копирования памяти.

Это не означает полное отсутствие взаимодействия с CPU:

- Python управляет выполнением; проверки/размеры/early stopping/метрики обмениваются скалярами.
  Полной CUPTI-трассировки внутренних передач библиотеки в этой задаче не делалось.
- Строковые категории остаются host sidecar: NumPy для их словарей и выбора строк обоснован.
  Аналогично host string idx; числовой Torch idx теперь не копируется на CPU для Python dict.
- Текущие hashlib/дисковый cache используют байтовое CPU-представление; это отдельная граница сериализации,
  а не вычисление числовых преобразований NumPy. Полный pipeline с cache не объявляется GPU-resident.
- CatBoost/LightGBM требуют прежние host library adapters; эта работа их API не меняет.
- gblinear не поддерживается inplace_predict: сохранён sklearn-путь. CPU TensorData при GPU model
  получает результат на CPU, поскольку это устройство запрашиваемого контейнера.

GPU output сохраняет best_iteration; labels/probabilities и их dtype согласованы с sklearn,
включая multilabel и multi:softmax. Для softmax допускается обычная погрешность вычислений CPU/GPU.

Проверки: 106 адресных CPU-тестов; 16 GPU-тестов, в том числе запрет cpu/numpy/tolist для row algorithms.
Industrial: порядок пересечения и смысл idx не менялись; источники нужны в соответствующих новых местах
при синхронизации. Подробности в записке Industrial.
