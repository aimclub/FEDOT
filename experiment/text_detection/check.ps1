# Run from the FEDOT root.
param([switch]$Full)

$ErrorActionPreference = 'Stop'

python -m compileall -q fedot/core/data/data_detection.py test/unit/data/test_text_detection_v2.py experiment/text_detection
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m pytest test/unit/data/test_improved_text_detector.py test/unit/data/test_text_detection_v2.py experiment/text_detection/tests -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m flake8 --config experiment/text_detection/.flake8 experiment/text_detection test/unit/data/test_text_detection_v2.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m mypy --config-file experiment/text_detection/mypy.ini experiment/text_detection
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Full) {
    python -m pytest test/unit/data -q
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
