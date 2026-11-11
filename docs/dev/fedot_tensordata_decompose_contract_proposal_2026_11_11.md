# Decompose на TensorData: выбранный контракт

11 ноября 2026. После обсуждения исходное предложение упрощено. Новых полей в TensorData нет:
main/auxiliary определяется графом, глубина вычисляется через GOLEM distance_to_primary_level.
Ни SupplementaryData, ни полная история целей, ни PredictionInfo в этом пункте не вводятся.
Runtime-изменения пока не закоммичены. Binary-дефект обсуждён и исправление сохранено.

## Назначение и сохранённая семантика

`decompose` передаёт признаки выбранного родителя и меняет обучающий target на y - baseline.
`class_decompose` вычитает вероятности из binary/one-hot target и переключает task на regression.
Далее обычная регрессионная модель учится предсказывать остатки. Финальная модель может объединять
базовые прогнозы и прогнозы остатков; сам decompose не задаёт финальное суммирование.

Исходный y хранится в основной ветви. Вспомогательная ветвь хранит свой остаточный target.
Если две модели разветвляются после одного decompose, их объединение продолжает учиться на остатке.
При возврате к основной ветви приоритет снова получает исходный y. Если все цели вспомогательные и
различаются, сохраняется прежняя эвристика: минимальная глубина, при равенстве — стабильный порядок.
Новый выбор более эффективной цели для ансамбля — отдельный эксперимент, не часть миграции.

## Разделение по файлам

- `core/pipelines/tensor_input_rules.py`: две pure функции для выбора источника target и ролей decompose.
  Их deterministic tests находятся в `tests/core/pipelines/test_tensor_input_rules.py`.
- `core/pipelines/tensor_input.py`: подготовка готовых parent outputs, с использованием общего merger.
- `core/pipelines/node.py`: получает родительские результаты в согласованном порядке и вызывает helper.
- `core/data/merge/data_merger.py`: принимает необязательный main_output_index, выравнивает строки;
  aligned_output возвращает одного родителя с его собственной схемой. Арифметики decompose и обхода
  графа внутри merger нет. Прямой вызов без graph context сохраняет прежний TensorData fallback.
- `core/operations/implementations/data_operations/decompose.py`: Torch fit/transform/transform_for_fit.
- `core/operations/strategies/data_operations/common_preprocessing.py`: тот же общий dispatcher,
  без отдельной стратегии ради двух одинаковых fit/transform контрактов. Старый preprocessing.py удалён.
- `core/pipelines/verification_rules.py`: проверка ролей использует тот же resolver; позиция Model в
  исходном nodes_from больше не имеет значения. Ровно два родителя и regression-потомок сохраняются.
- `core/caching/fingerprints.py`: predict участвует в identity, когда служит baseline входа decompose.

## Выбор цели по графу

`resolve_tensor_target_source(parents, has_target=None)`:

1. Decompose — граница вспомогательной цели. Первичные узлы начинают основную цель.
2. Другой узел наследует основную цель, если она доступна хотя бы через одного родителя;
   если все родители вспомогательные, его цель тоже вспомогательная.
3. Среди основных родителей выбирается первый с target, если labels доступны. Это сохраняет обычный
   TensorData-сценарий, когда labels есть только в одной модальности. Если labels нет, сохраняется
   первый основной источник task. Наличие labels у auxiliary не повышает его приоритет.
4. Если все родители auxiliary, выбирается минимальная graph depth, затем стабильный порядок.

Рекурсивное определение использует memo только внутри вызова. Изменение графа не оставляет старую
роль в данных/долговечном кеше. При inference выбор основной задачи не зависит от наличия labels.
Граф определяет источник, но сами значения target берутся из актуального результата выбранной ветви.
Из исходного входа заново y не подставляется: это разрушило бы обучение ансамбля остатков.

## Выбор двух входов decompose

`resolve_decompose_parents(parents)`:

- требуется ровно два родителя, хотя бы один Model;
- Model + DataOperation: прогноз берётся у Model, признаки — у DataOperation;
- Model + Model: более глубокая модель даёт baseline; при равенстве — первая в execution order;
- второй Model в роли признаков даёт свои predictions как числовые features.

Это сохраняет рабочие legacy-графы с двумя моделями. Явная роль Model/DataOperation устраняет ошибочное
угадывание по глубине, когда цепочка data operations оказалась длиннее модельной ветви.

Parent outputs выравниваются по общим idx через существующие Torch helpers. Подготовленный TensorData:

```python
features = aligned_feature_parent.semantic_output
predict = aligned_prediction_parent.predict
target = aligned_selected_target
task = selected_target_source.task
```

Для DataOperation сохраняются его names, numeric/categorical indices, source mapping и raw categories.
Для Model-as-features схема становится числовой. Склеивать X с baseline, а потом резать обратно не нужно.
Output data operation очищает predict; target заменяется только в новом контейнере. Соседняя ветвь
не теряет свой y. На inference остатки не вычисляются. Если caller передал y, он сохраняется как в
legacy transform; предложение автоматически очищать его в auxiliary-ветви сейчас не реализовано.

## Математика и формат

- Regression vectors приводятся к (N,1), multioutput сохраняется как (N,T); размерности должны совпадать.
- Classification сохраняет отсортированный словарь классов в fitted decomposer при fit.
- Binary использует класс с большим значением как положительный; метки не обязаны быть 0/1.
- Binary full_probs берёт положительную колонку и возвращает один residual target.
- Multiclass строит target matrix сравнением с fitted vocabulary и вычитает probabilities.
- Binary multilabel выдаёт по одному остатку на target, в том числе при probabilities (N,T,2).
- После fit vocabulary не сужается на последующем subset. Если уже при начальном fit в y нет класса,
  который присутствует в колонках producer, возникает явная ошибка несовпадения осей. Получение полного
  producer class order и общий multioutput multiclass/ragged контракт остаются отдельными задачами.
- Здесь не добавляется автоматическое распознавание labels вместо probabilities. Pipeline передаёт
  стандартный classification probability output; прямой caller обязан соблюдать этот входной контракт.

Все числовые операции — Torch на устройстве входа. Host categories/string idx используют прежний
обоснованный host helper. Сложные GPU модели и предпочтение устройств к этой арифметике не относятся.

## Baseline и обнаруженный binary-дефект

До изменений сохранены 16 InputData cases: regression/multioutput/binary/multiclass, разные порядки
родителей и разные/equal depths. Артефакты: /tmp/fedot_decompose_before.py,
/tmp/fedot_decompose_baseline.py, /tmp/fedot_decompose_baseline.npz, /tmp/fedot_compare_decompose.py.

В legacy `_binary_difference` использовалось `bin_target[np.argwhere(...)]`. Для column target argwhere
возвращает пары (row,0), а это выражение индексирует ими только ось строк. Нулевая строка попадала в
оба присваивания, и после второго её значение всегда становилось 1. При отрицательном первом объекте
остаток был завышен на 1. Torch comparison даёт правильную метку. Исправление обсуждено с пользователем
и сохранено: знак остатка означает over/underprediction положительного класса. Следующий регрессор
получает именно signed numeric target, без clipping и повторной классификационной интерпретации.
Прямой тест на двух объектах обучает linear на [-0.2,0.8] и проверяет такие же предсказания.
12 regression/multiclass baseline cases совпали точно; 4 binary cases отличаются только первой negative row,
остальные строки совпали точно. Финальный адресный CPU-run: 82 passed, 2 skipped, 1 unrelated deselected;
2 реальных CUDA-проверки прошли отдельно. Несвязанный legacy test multiple roots ожидает старый текст
исключения вместо structured validation error, root validation в этой задаче не менялся.

Полная regression-схема scaling → ridge → decompose → linear → ridge сравнена с прежним InputData
merger и замороженной реализацией: max prediction error 1.42e-14, target точно сохранён.
Скрипт: /tmp/fedot_decompose_pipeline_parity.py.

## Кеш и существующие пути

Поскольку baseline находится в predict, cache identity теперь учитывает sampled prediction rows,
их shape/dtype. Семплирование predict отдельно от features учитывает разные layouts, включая 3D features
с одним prediction на sample. После подготовки/изменения features/target fingerprint сбрасывается.
Hashing остаётся ответственностью cache layer, не математической операции.

Обычный tabular merge обновляет схему даже без categorical sidecar: склеенные predictions не должны
наследовать numeric indices исходной X. Прежний multidimensional TS/image metadata layout сохранён.
Новые InputData bridges и поддержка старых обученных pipeline files не добавляются.
Native OOF/TS horizon semantics здесь не реализуются. Старые train-vs-OOF predictions не подменяются.

Industrial переиспользует decompose implementation, но его отдельный legacy-merger не получает graph
context автоматически. При синхронизации использовать аналогичные границы target и подготовку родителей;
не возвращаться к выбору любого непустого target. Внешний Industrial не изменялся.
