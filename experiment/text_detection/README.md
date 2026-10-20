# Text-column detection benchmark

The current text detector is implemented in `fedot/core/data/data_detection.py`.
This directory contains the benchmark, the previous detector, and two
historical baselines. Only the current detector is used by FEDOT itself.

The historical baselines are approximations, not exact copies of the original
FEDOT implementation or PR #903.

## Run

Clone `sxjscience/automl_multimodal_benchmark` next to FEDOT:

```text
auto_ml_lab/
    FEDOT/
    automl_multimodal_benchmark/
```

From the FEDOT repository root:

```powershell
python -m experiment.text_detection.benchmark
python -m experiment.text_detection.diagnostics
python -m experiment.text_detection.benchmark --split test
```

Results are written to `reports/` and ignored by Git. To work with the results,
open `results.ipynb`. The `test` split has a separate `reports/test/` directory.

The runner fails if a dataset cannot be loaded or has unlabelled columns.
`--allow-partial` can be used to debug an incomplete run.

## Labels

Manually reviewed annotations live in
`labels/benchmark_columns_to_label.csv`. Labels mean *semantic text potentially
useful for NLP*, not all string columns. Some benchmark column names begin with
spaces; they must be matched without stripping those spaces.

Datasets providing their own `feature_types` are evaluated using those labels.
Other datasets use the CSV. Annotations were partly created manually and may
need further validation.

## Checks

```powershell
python -m pytest test/unit/data/test_improved_text_detector.py -q
python -m pytest test/unit/data/test_text_detection_v2.py -q
python -m pytest experiment/text_detection/tests -q
python -m pytest test/unit/data -q
python -m flake8 --config experiment/text_detection/.flake8 experiment/text_detection
python -m mypy --config-file experiment/text_detection/mypy.ini experiment/text_detection
```

This benchmark was also used during the development of the detector. Its scores
are exploratory, not an independent estimate of generalization. A test split
from the same datasets does not replace evaluation on new datasets.
