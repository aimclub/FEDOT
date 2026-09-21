# FED-02: контракт внешних операций

## Границы изменения

`fedot.extensions` остаётся единственной точкой регистрации расширений. Встроенный
каталог операций не подменяется и не дополняется глобальными записями расширений.
`OperationFactory` выбирает адаптер по спецификации из текущей области регистрации.
Изменения не затрагивают подготовку TensorData из FED-01, эволюционный поиск,
Industrial или GOLEM. В `Pipeline` добавлена только передача контекста в поток
выполнения с ограничением времени.

## Модели и преобразования

`ExtensionManifest` сохраняет прежние позиционные аргументы. Поле `transforms`
добавлено в конец; `models` может быть пустым, если объявлено хотя бы одно
преобразование. `ExternalModelSpec`, `ModelCapabilities`,
`ModelHyperparamsSchema` и обнаружение `FEDOT_EXTENSION_MANIFEST` сохранены.

| Контракт | Обучение | Выполнение | Результат |
| --- | --- | --- | --- |
| `ExternalModelSpec` + `ModelCapabilities` | `fit` | `predict` или `predict_proba` | `ModelOutput.prediction` → `TensorData.predict` |
| `ExternalTransformSpec` + `TransformCapabilities` | `fit`, если `requires_fit=True` | `transform` | `TransformOutput.features` → `TensorData.features` |

Входы интерпретатора представлены отдельными типами `ModelInput` и
`TransformInput`. Протоколы `ModelImplementation` и `TransformImplementation`
описывают основные интерфейсы. Преобразование без обучения объявляет
`requires_fit=False` и может не иметь метода `fit`.

Оба набора возможностей объявляют задачи, типы данных, теги, способ представления
массивов и необходимость целевой переменной. Модель по умолчанию требует цель
при обучении; для обучения без учителя нужно указать `requires_target=False`.
Преобразование обязано явно указать `output_data_type`; модель при отсутствии
этого поля сохраняет тип входных данных. Методы модели больше не подменяются
методом `transform`: такие расширения нужно перенести в `transforms`.

`ArrayBackend.numpy` является значением по умолчанию для совместимости с
классическими оценивателями. `ArrayBackend.torch` передаёт тензоры. Адаптер
TensorData создаёт отдельные входные буферы для внешнего кода, возвращает результат
как тензор на устройстве входных признаков и проверяет число строк. Преобразование
очищает `predict` и устаревшие описания столбцов, но сохраняет индексы строк и цель.
Изменение числа строк этим контрактом не поддерживается.

## Область регистрации

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.extensions import (
    ExtensionManifest, ExternalModelSpec, ExternalTransformSpec,
    ModelCapabilities, TransformCapabilities, extension_scope,
)

manifest = ExtensionManifest(
    name='example', version='1',
    models=(ExternalModelSpec(
        'example_linear', LinearRegression,
        ModelCapabilities((TaskTypesEnum.regression,), (DataTypesEnum.tabular,)),
    ),),
    transforms=(ExternalTransformSpec(
        'example_scaler', StandardScaler,
        TransformCapabilities(
            (TaskTypesEnum.regression,), (DataTypesEnum.tabular,), DataTypesEnum.tabular,
        ),
    ),),
)

with extension_scope(manifest):
    # Build, fit and predict the pipeline within this scope.
    ...
```

`extension_scope(*manifests)` наследует видимые записи и восстанавливает прежнее
состояние в `finally`, в том числе при ошибке входа или выполнения тела. Нельзя
переопределить родительскую запись. Имена моделей и преобразований принадлежат
одному пространству имён; также проверяются конфликты с активным встроенным
каталогом. Суффикс узла `/name` допустим при использовании операции, но не в её
объявлении.

`register_extensions(manifests)` сначала проверяет весь пакет и лишь затем
публикует его одной заменой состояния. Конфликт или неверный манифест не оставляют
частичных записей. `register_extension` сохраняет прежний результат
`Right(RegisteredExtension)`; пакетная операция возвращает кортеж таких записей.
Вне контекстного менеджера регистрация действует до явной очистки текущего контекста.

Хранилище использует `ContextVar` с неизменяемым кортежем записей. Копии контекста
не изменяют состояние друг друга. Новый поток или процесс не наследует область
автоматически: для потока нужен явный перенос через `copy_context().run`, для
процесса нужна регистрация манифеста в этом процессе. `Pipeline.fit` с
`time_constraint` переносит контекст в поток исполнителя; оба варианта выполнения
проверяются одним интеграционным тестом. Объекты манифестов и вложенные значения
параметров после регистрации следует считать неизменяемыми.

## Проверка и планы

Чистые правила расположены рядом с точками входа:

- `validation.py`: форма манифеста, разные типы спецификаций и возможностей,
  схема параметров и доступная сигнатура фабрики.
- `registration_rules.py`: `RegistrationPlan` с именами расширений, операций и
  их видами; план не хранит фабрики, оцениватели или реестр.
- `call_rules.py`: `CallPlan` с номером совместимого варианта вызова.
- `execution_rules.py`: `ExecutionPlan` с видом операции, представлением массивов
  и выходным типом данных; проверка задачи, входного типа и наличия цели.
- `parameter_rules.py`: объединение значений по умолчанию с параметрами
  пользователя и проверка обязательных и неизвестных ключей.

`register_extensions(..., dry_run=True)` возвращает план и не меняет реестр.
Проверка манифеста, параметров и построение планов не создают оценивателей и не
вызывают их методы. `discover_extensions` отдельно импортирует указанные модули:
импорт Python по своей природе может выполнять код модуля и не является чистой
проверкой.

`smoke_test_extension(manifest, parameters=None)` остаётся проверкой фабрик,
а не конвейера. `parameters` сопоставляет имя операции с её параметрами. Сначала
проверяются параметры всех операций, затем создаются экземпляры. Отдельный
интеграционный тест действительно обучает и выполняет двухузловой конвейер
TensorData на CPU.

## Сигнатуры и ошибки

Совместимость определяется через `inspect.signature(...).bind(...)`, а не
`bind_partial`. Фабрика поддерживает варианты `factory(params)`,
`factory(params=params)` и `factory()` в таком порядке. Методы обучения поддерживают
`fit(features, target)`, `fit(features)`, `fit(idx, features, target, params)` и
`fit(idx, features, target)`. Методы предсказания и преобразования поддерживают
`method(features)`, `method(idx, features, params)` и `method(idx, features)`.
При неоднозначной сигнатуре выбирается первый полностью связанный вариант.

После выбора выполняется ровно один вызов. Исключение из тела функции не означает
несовместимость сигнатуры и не вызывает повтор. `ExtensionContractError` хранит
`error: ExtensionError` и `code`; исходное исключение доступно через `__cause__`
и `error.cause`. При возврате `Left` причина сохраняется в `ExtensionError.cause`.

Сохранены прежние коды, в частности `invalid_manifest_type`, `empty_models`,
`duplicate_extension`, `duplicate_model_name`, `invalid_factory_signature`,
`factory_smoke_test_failed`, `factory_returned_none`,
`missing_required_hyperparams` и `validation_error`. Код `empty_models` теперь
означает отсутствие любых операций, а `duplicate_model_name` применяется и к
повтору имени преобразования внутри манифеста.

Новые устойчивые коды: `operation_name_conflict`, `invalid_capabilities`,
`invalid_hyperparams_schema`, `invalid_parameters`, `operation_not_registered`,
`unsupported_method_signature`, `missing_runtime_method`,
`extension_execution_failed`, `unsupported_task`, `unsupported_data_type`,
`target_required`, `unsupported_output_mode`, `invalid_runtime_output`,
`output_row_mismatch`. Ключ со значением `None` не считается отсутствующим;
явный `None` в значениях по умолчанию также сохраняется.

## Проверки и ограничения

Тесты охватывают атомарную регистрацию, вложенные области и откат, конфликты между
моделями, преобразованиями и встроенными операциями, изоляцию копий контекста,
обнаружение манифеста, предварительную проверку параметров, выбор сигнатуры,
сохранение причины `TypeError`, ошибки результата и отсутствие повторных вызовов.
Конечные наборы перестановок проверяют независимость поиска от порядка регистрации,
сохранение конфликта при добавлении записей и идемпотентность разрешения параметров.
Эти проверки используют pytest и не добавляют зависимость от Hypothesis.

Расширенный прогон выявил две несогласованности в тестах и совместимом интерфейсе
базы `4104a16e`. В проверке PCA срез признаков теперь сопровождается тем же срезом
индексов и цели. Цепочный интерфейс
`PipelineOperationRepository.from_available_operations` снова возвращает `self`,
как уже закреплено тестом репозитория.

Интеграционный тест `test_real_cpu_tensor_pipeline_model_and_transform` использует
настоящие `StandardScaler` и `LinearRegression`. Он проверяет прогноз, индексы,
неизменность входных признаков и точную последовательность вызовов. Repository
не подменяется через monkeypatch. Дополнительно проверяется преобразование
тензоров без обучения.

GPU, сторонние многопоточные исполнители, сериализация обученных расширений
и многомодальный запуск не входят в эту проверку. Признак
`supports_multimodal` сохранён как метаданные, но сам по себе не добавляет такой
режим в адаптер TensorData. Схема параметров проверяет имена и наличие значений,
но не задаёт типы и числовые диапазоны отдельных гиперпараметров.

Команды проверки из корня рабочей копии:

```text
python -m pytest tests/extensions tests/core/operations/test_operation_factory_extensions.py tests/core/operations/test_extension_tensor_pipeline.py tests/core/repository/test_operation_types_repository_extensions.py -q
python -m py_compile <changed Python files>
python -m autopep8 --max-line-length 120 --diff <changed Python files>
git diff --check
```
