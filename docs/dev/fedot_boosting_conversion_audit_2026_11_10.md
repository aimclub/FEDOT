# Общие преобразования массивов и бустинги

10 ноября 2026. Обновлено после выноса конвертеров. Номера строк относятся к текущему рабочему дереву.
Классификация/регрессия каждого семейства используют одинаковые пути обмена массивами.

## 1. Общий модуль

[core/backend/array_conversions.py](../../fedot/core/backend/array_conversions.py) не читает global Backend.

| Функция | Контракт | Где происходит фактическая конверсия |
| --- | --- | --- |
| [as_numpy:13](../../fedot/core/backend/array_conversions.py#L13) | CPU NumPy; dtype сохраняется, если явно не задан; copy=True даёт независимое хранилище | Torch: detach/cpu/numpy на строке 18; CuPy: get на 21; NumPy dtype/copy на 25 |
| [as_torch:28](../../fedot/core/backend/array_conversions.py#L28) | Torch; исходное устройство сохраняется, host sequences/NumPy остаются CPU, пока явно не задано иное | CuPy DLPack на 42; NumPy → Torch на 44; device/dtype/copy через Tensor.to на 45; contiguous на 46 |
| [as_cupy:49](../../fedot/core/backend/array_conversions.py#L49) | CUDA array; device/dtype/copy/contiguous задаются явно при необходимости | Torch DLPack на 70; другие массивы через cupy.array на 73 |
| [torch_to_xp:77](../../fedot/core/backend/array_conversions.py#L77) | Существующий preprocessing-вызов с явным NumPy/CuPy module | Теперь делегирует первым трём функциям; перенесён из backend.py вместе с импортами потребителей |

По умолчанию общего скрытого float64, приватной копии или выбора устройства по Backend нет.
Torch inputs отсоединяются от autograd для внешних библиотек; detach=False сохраняет граф, когда это нужно
контракту extensions. dtype у as_torch — torch.dtype, у as_numpy/as_cupy — dtype соответствующей библиотеки.
CuPy импортируется внутри as_cupy как optional GPU dependency; CPU-only вызовы не загружают его.

CPU NumPy ↔ Torch и CUDA CuPy ↔ Torch разделяют память при подходящих dtype/layout/device и copy=False.
Смена устройства, dtype, требование contiguous или приватной копии могут выделять новый буфер.

## 2. Что стало с _to_model_device и _prediction_tensor

До выноса _to_model_device выбирал XGBoost device, затем выполнял detach → to(device) → contiguous.
Он использовался для features/target на fit, eval set и для features на predict.
Теперь функция удалена: в XGBoost осталась только [_model_device:103](../../fedot/core/operations/implementations/models/boosting.py#L103)
с библиотечной политикой device='cpu'/'cuda:N'. Универсальный перенос выполняет as_torch.

_prediction_tensor переименован в [_prepare_prediction:87](../../fedot/core/operations/implementations/models/boosting.py#L87).
Он больше не содержит низкоуровневых преобразований: вызывает as_torch. Локально остаётся упаковка
списка LightGBM multioutput через stack(dim=1), чтобы получить samples × targets × classes.
Эта ось — контракт модели, а не общий способ превращения любого Python list в tensor.

## 3. Основные строки после изменений

Все ссылки в таблице — на [implementations/models/boosting.py](../../fedot/core/operations/implementations/models/boosting.py).
«Раньше» ниже означает старый InputData-код `3c1f4f3d`, до текущего переноса семейства.

| Семейство / этап | Раньше | Текущее выражение и строка |
| --- | --- | --- |
| XGBoost features | NumPy → pandas DataFrame | [111](../../fedot/core/operations/implementations/models/boosting.py#L111): as_torch(features, device=self._model_device, contiguous=True); остаётся Torch |
| XGBoost target | NumPy → pandas DataFrame | [114](../../fedot/core/operations/implementations/models/boosting.py#L114): as_torch(self._target_tensor(data), device=self._model_device, contiguous=True) |
| XGBoost CUDA output | sklearn возвращал CPU NumPy | [128](../../fedot/core/operations/implementations/models/boosting.py#L128): as_torch(prediction), где prediction — CuPy от Booster.inplace_predict; внутри DLPack |
| XGBoost готовый output | OutputData | [144](../../fedot/core/operations/implementations/models/boosting.py#L144): as_torch(prediction, device=data.device) |
| XGBoost CPU / gblinear output | NumPy → OutputData | [120](../../fedot/core/operations/implementations/models/boosting.py#L120) → общий prepare_prediction, [90](../../fedot/core/operations/implementations/models/boosting.py#L90): as_torch(prediction, device=data.device) |
| LightGBM features | NumPy → pandas DataFrame | [194](../../fedot/core/operations/implementations/models/boosting.py#L194): as_numpy(self._feature_tensor(data, fit=fit)) |
| LightGBM/CatBoost target | NumPy, у LightGBM ещё DataFrame | [84](../../fedot/core/operations/implementations/models/boosting.py#L84): as_numpy(self._target_tensor(data)) |
| LightGBM/CatBoost output | NumPy → OutputData | [89–90](../../fedot/core/operations/implementations/models/boosting.py#L89): as_torch; список LightGBM собирается через stack |
| CatBoost числовой блок с категориями | Общая NumPy-матрица → Pool | [249](../../fedot/core/operations/implementations/models/boosting.py#L249): as_numpy(original.numerical_features, dtype=np.float32) |
| CatBoost features без исходных категорий | NumPy непосредственно в Pool | [259](../../fedot/core/operations/implementations/models/boosting.py#L259): as_numpy(data.features) |
| CatBoost категории и Pool | Исходные значения в NumPy → Pool с cat_features; FeaturesData не было | [244](../../fedot/core/operations/implementations/models/boosting.py#L244): astype(str).astype(object); [253](../../fedot/core/operations/implementations/models/boosting.py#L253): FeaturesData; [261](../../fedot/core/operations/implementations/models/boosting.py#L261): Pool |
| XGBoost/LightGBM коды категорий | pandas dtype category | [75](../../fedot/core/operations/implementations/models/boosting.py#L75): as_torch(encoded, device=data.device, dtype=data.features.dtype) |

Pool/FeaturesData, строки категорий, LightGBM categorical_feature, XGBoost objective/best_iteration,
порог labels/argmax и требуемые dtype меток остались в моделях. Числовые признаки из GPU не извлекаются
ради кодировщика: исходные строки уже хранятся в NumPy sidecar, на GPU поступают числовые коды.

## 4. Другие потребители и сохранённые контракты

- cuML: features/target → as_torch с модельным dtype/device, затем as_cupy; output → as_torch.
- TensorSkLearn и neural tabular: одинаковые общие вызовы вместо локальных cpu/numpy/as_tensor.
  У neural удалён дублирующий _to_numpy; их preprocessing и алгоритмы не переписывались.
- Extensions: as_numpy/as_torch(copy=True), для Torch также detach=False — сохранены private storage и autograd.
- Source/preparation/obligatory preprocessing/embedding: torch_to_xp импортируется из общего модуля.
- Host sidecar, merger, InputData bridge, hashing, legacy OOF bridge и routing используют те же функции.
- `core/common/tools.py` удалён: там был единственный дублирующий to_numpy. Fingerprints используют as_numpy.
- `tensor_data.tools.to_tensor` сохранён как политика создания данных: приводит подготовленные значения
  (в том числе numeric object arrays после кодирования) к float64, затем вызывает as_torch с запрошенным dtype,
  Backend.device и copy=True. Старые defaults/ownership сохранены; это не общий converter для моделей.
- TensorData.to(device) остаётся перемещением целого контейнера; выбор compute device не перенесён в конвертеры.
- Скаляры loss/метрик и алгоритмические dtype-приведения не являются обменом форматов массивов и оставлены локально.

Низкоуровневые cpu().numpy(), Torch/CuPy from_dlpack и torch.as_tensor в FEDOT core теперь находятся
в array_conversions.py. Встроенный Industrial и экспериментальные legacy-baselines этим проходом не переписывались.

## 5. Файлы текущего выноса

Новый `fedot/core/backend/array_conversions.py`; удалён `fedot/core/common/tools.py`.

Изменены:

- `fedot/core/backend/backend.py`
- `fedot/core/caching/fingerprints.py`
- `fedot/core/data/bridges/tensor_to_input_rules.py`
- `fedot/core/data/merge/data_merger.py`
- `fedot/core/data/tensor_data/contracts.py`
- `fedot/core/data/tensor_data/preparation.py`
- `fedot/core/data/tensor_data/row_operations.py`
- `fedot/core/data/tensor_data/source_layout.py`
- `fedot/core/data/tensor_data/tools.py`
- `fedot/core/operations/implementations/models/boosting.py`
- `fedot/core/operations/implementations/models/neural.py`
- `fedot/core/operations/strategies/extensions.py`
- `fedot/core/operations/strategies/models/cuml.py`
- `fedot/core/operations/strategies/models/sklearn.py`
- `fedot/core/optimisers/objective/oof_objective_rules.py`
- `fedot/core/pipelines/ensembling/routing.py`
- `fedot/preprocessing/methods/embedding.py`
- `fedot/preprocessing/service/obligatory_service.py`

Добавлен `tests/core/backend/test_array_conversions.py`.
Документы: этот аудит, общие требования migration order и записка синхронизации Industrial.
Прежние изменения рабочего дерева сохранены; список выше относится только к этому выносу.

Проверки: 187 адресных CPU-проверок и 29 GPU-проверок; дополнительно короткий CPU fit/predict gating router.
Проверены shared pointers, dtype, explicit copy, contiguous, None, autograd, XGBoost, cuML,
neural/sklearn, создание данных, extensions, bridge и hashing. Единственный сетевой loader-тест
успешно повторён вне sandbox после DNS-ошибки внутри него. Полного AutoML поиска/бенчмарка повторно не запускали.

## 6. CPU/GPU выбор не изменён

У бустингов одна FEDOT implementation на семейство. GPU задаётся параметрами библиотеки:
XGBoost device='cuda:0' (обычно tree_method='hist'), CatBoost task_type='GPU'/devices='0',
LightGBM device_type='gpu'. По умолчанию CPU. CUDA TensorData само по себе не включает GPU-обучение.
XGBoost _model_device выбирает устройство; as_torch только выполняет запрос, не выбирая предпочтения.
Глобальная автоматическая политика остаётся отдельной задачей из CPU/GPU checklist.
