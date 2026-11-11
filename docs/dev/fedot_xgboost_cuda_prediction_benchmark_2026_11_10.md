# Мини-бенчмарк CUDA prediction XGBoost

10 ноября 2026. Сравнение текущего FEDOT inference с экспериментальным GPU output-путём.
Реализации моделей не изменялись; кандидат пока находится только в benchmark-скрипте.

## Результат

Для batch 250 000 строк прямой GPU-путь быстрее в 1,16–2,38 раза на проверенных задачах/режимах.
Для batch 256 строк экономия только 0,04–0,10 мс; маленькие различия могут зависеть от шума замеров.
Все 15 сравнений значений/форм/dtype между путями совпали точно, rtol=atol=0.
Это измерение inference, а не ускорения самого алгоритма обучения или полного AutoML.

## Что сравнивалось

- Текущий путь: FEDOT implementation.predict/predict_proba → sklearn XGBoost → NumPy на CPU → Torch на CUDA.
- Кандидат: те же Torch features → Booster.inplace_predict → CuPy на CUDA → torch.from_dlpack.
  Для binary probabilities [1-p, p], labels и multiclass argmax обработка выполняется Torch на GPU.
- Одинаковые обученные модели и CUDA-входы; одинаковый диапазон до best_iteration включительно.
  Используется та же missing-value настройка. Candidate ограничен стандартными regression,
  binary:logistic и multi:softprob; универсальную замену inference он не реализует.
- Замеряется полный вызов реализации до готового Torch-результата. TensorData-упаковка стратегии не включена
  ни в один вариант. Преобразование входов и библиотечная Python-обвязка входят в оба пути.
- Три прогрева каждого варианта, девять измерений, порядок вариантов перемешан с фиксированным seed.
  perf_counter + torch.cuda.synchronize до/после каждого измерения; ниже медианы, исходные повторы в JSON.
  Экономия включает исключение пересылок, sklearn-обвязки и CPU postprocessing, а не только PCIe-копии.

## Окружение и модели

RTX 4080 SUPER; Python 3.10.20; Torch 2.7.1+cu126; CUDA runtime Torch 12.6;
XGBoost 3.0.5; CuPy 13.5.1. Torch threads=2, XGBoost nthread=2.
Синтетические 12 000 × 32 числовых train features float32, генерация сразу на CUDA, seed=17.
Внутренний eval set через текущий FEDOT split, patience=15, максимум 100 boosting rounds, depth=6.
Inference features генерируются отдельно и уже находятся на CUDA.

| Задача | Выполнено rounds | Используется при predict (best+1) | Время единственного fit, мс |
| --- | ---: | ---: | ---: |
| Regression | 37 | 22 | 142,94 |
| Binary | 46 | 31 | 111,86 |
| Multiclass, 5 классов | 100 | 100 | 568,56 |

Время fit дано только как описание масштаба, без сравнения двух вариантов обучения.
Оба варианта prediction используют одну и ту же обученную модель.

## Замеры

| Задача / результат | Batch | Через NumPy, мс | CuPy → Torch, мс | Ускорение |
| --- | ---: | ---: | ---: | ---: |
| Regression | 256 | 0,637 | 0,559 | 1,14× |
| Regression | 10 000 | 0,637 | 0,560 | 1,14× |
| Regression | 250 000 | 1,411 | 1,015 | 1,39× |
| Binary probabilities | 256 | 0,680 | 0,633 | 1,07× |
| Binary labels | 256 | 0,724 | 0,626 | 1,16× |
| Binary probabilities | 10 000 | 0,722 | 0,648 | 1,11× |
| Binary labels | 10 000 | 0,788 | 0,652 | 1,21× |
| Binary probabilities | 250 000 | 1,641 | 1,079 | 1,52× |
| Binary labels | 250 000 | 2,586 | 1,085 | 2,38× |
| Multiclass probabilities | 256 | 2,636 | 2,591 | 1,02× |
| Multiclass labels | 256 | 2,677 | 2,590 | 1,03× |
| Multiclass probabilities | 10 000 | 2,697 | 2,544 | 1,06× |
| Multiclass labels | 10 000 | 2,877 | 2,599 | 1,11× |
| Multiclass probabilities | 250 000 | 8,618 | 7,436 | 1,16× |
| Multiclass labels | 250 000 | 12,125 | 7,535 | 1,61× |

На labels выигрыш больше также за счёт переноса порога/argmax с CPU на GPU.
Это один небольшой синтетический эксперимент на одной GPU, не общее обещание ускорения.
Multioutput, multi:softmax, custom objective, dart/gblinear и CPU TensorData с GPU model здесь не измерялись.
Перед основной реализацией нужно сохранить их текущие контракты; менять CPU-путь ради этого замера не требуется.

## Остались ли CPU-переходы при обучении

У XGBoost числовые features и target передаются в native fit как CUDA Torch tensors.
На нашей границе модели копии всей числовой features-матрицы на CPU нет.
Но весь FEDOT fit не является полностью GPU-resident:

- Стратификация внутреннего eval set копирует target на CPU и использует sklearn/CPU row positions.
  Срезы features затем выполняются Torch на их устройстве.
- Исходные строки категорий и их словари сейчас находятся на CPU; benchmark содержит только числа.
- Python управляет fit/early stopping, получает метрики и скалярные результаты проверок с GPU.
  Это отличается по объёму от пересылки всей features-матрицы или prediction batch.
- Создание исходного TensorData и прочий preprocessing в этот замер не входят.

Эти пункты установлены по коду границ и split; полной CUPTI-трассировки всех внутренних копий XGBoost не было.
Нельзя утверждать полное отсутствие CPU-обмена внутри библиотеки.
CatBoost/LightGBM в текущих адаптерах по-прежнему получают host input и не охвачены этим ускорением.

## Воспроизведение

Скрипт: [xgboost_cuda_prediction.py](../../examples/benchmark/xgboost_cuda_prediction.py).
Машинные результаты с каждым повтором: [JSON](fedot_xgboost_cuda_prediction_benchmark_2026_11_10.json).

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m examples.benchmark.xgboost_cuda_prediction \
  --output /tmp/xgboost_cuda_prediction.json
```

Практический вывод: GPU output через DLPack имеет смысл реализовать с сохранением estimator semantics;
текущий NumPy roundtrip на CUDA не обязателен. Автоматические CPU/GPU preferences этим замером не определяются.
