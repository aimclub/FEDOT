# GPU build сценарий для TensorData benchmark на Windows

## Зачем нужен отдельный сценарий

Для `run_tabular_tensor_suite` на Windows у нас наложились две проблемы:
- старая `.venv_torch` была привязана к отсутствующему Python 3.9
- установленный `torch` wheel был `cp39`, а Windows torch build дополнительно требовал `mkl`, из-за чего импорт падал на `shm.dll` / `torch_cpu.dll`

Для benchmark use case безопаснее не чинить старое окружение по месту, а пересобирать отдельную benchmark-oriented GPU env.

## Что делает новый сценарий

Файлы:
- `examples/benchmark/setup_tensordata_gpu_env.ps1`
- `examples/benchmark/requirements_tensordata_gpu_windows.txt`

Сценарий:
- подбирает Python `3.10 -> 3.9 -> 3.8` через `py`, либо принимает явный `-PythonExe`
- пересоздаёт `.venv_torch` при `-Recreate`
- ставит минимальный benchmark dependency set для Windows без Linux-first extras вроде `cudf-cu12`
- ставит локальный `fedot` в editable mode без повторной установки полного `requirements.txt`
- ставит `torch==2.3.1+cu121`, `torchvision==0.18.1+cu121`, `torchaudio==2.3.1+cu121`
- явно ставит `mkl<=2021.4.0,>=2021.1.1`, чтобы закрыть Windows DLL chain для `torch`
- опционально ставит `dask/distributed/dask-ml`
- проверяет `import torch`, `torch.cuda.is_available()` и `from fedot.api.main import Fedot`
- опционально запускает GPU smoke benchmark

## Базовый запуск

Из корня репозитория:

```powershell
powershell -ExecutionPolicy Bypass -File .\examples\benchmark\setup_tensordata_gpu_env.ps1 -Recreate
```

С Dask и smoke-run:

```powershell
powershell -ExecutionPolicy Bypass -File .\examples\benchmark\setup_tensordata_gpu_env.ps1 -Recreate -InstallDask -RunSmoke
```

Если нужен конкретный интерпретатор:

```powershell
powershell -ExecutionPolicy Bypass -File .\examples\benchmark\setup_tensordata_gpu_env.ps1 -Recreate -PythonExe "C:\Program Files\Python38\python.exe"
```

## Что ожидаем после успешной сборки

Проверка должна пройти на трёх уровнях:
1. `torch` импортируется без `WinError 126`
2. `torch.cuda.is_available()` возвращает `True` на GPU-машине
3. `Fedot` импортируется без ошибки по `golem`

При `-RunSmoke` ожидается запуск:

```powershell
.\.venv_torch\Scripts\python.exe .\examples\benchmark\run_tabular_tensor_suite.py --datasets kc2 --modes input_gpu_bridge,tensor_gpu_bridge --seeds 42 --generations 1 --pop-size 3 --output-dir benchmark_results/tabular_tensor_suite_gpu_smoke
```

## Ограничения и замечания

- Сценарий ориентирован именно на текущий benchmark vertical slice, а не на полный industrial dev env.
- Мы специально не тянем сюда `cudf-cu12`, `cupy-cuda12x` и `sentence-transformers`, потому что они не нужны для текущего tabular TensorData benchmark и часто осложняют Windows setup.
- Если после пересборки `torch.cuda.is_available()` остаётся `False`, проблема уже не в wheel/venv, а обычно в драйвере NVIDIA, CUDA runtime compatibility или недоступной GPU в текущей сессии.
- Если нужен только CPU smoke-run, лучше использовать отдельный CPU сценарий и не ставить CUDA build.