# TensorData Vertical Slice PR Map

## Included scope

This PR-ready slice packages the current TensorData-first work completed across Slices A, B, and the first part of Slice C.

Included areas:
- TensorData taxonomy, normalization, creator resolution, backend planning, and file-source rules
- compatibility mapping between legacy `InputData` data types and TensorData canonical types
- `InputData -> TensorData` and `TensorData -> InputData` bridges
- bridge-aware entrypoints in `ApiDataProcessor` and `DataPreprocessor`
- explicit TensorData runtime entrypoints in `Pipeline`
- explicit TensorData inference and predefined-fit entrypoints in `Fedot`
- mirrored tests in `tests/core/data/...`, `tests/preprocessing/...`, `tests/core/pipelines/...`, and `tests/api/...`

## Explicit non-goals

Not included in this slice:
- full composer support for TensorData
- generic TensorData composition or `predefined_model='auto'` fit path
- industrial convergence
- broad rewrite of the legacy `InputData` API surface

## Review order

1. `fedot/core/data/`
2. `fedot/preprocessing/`
3. `fedot/core/pipelines/`
4. `fedot/api/api_utils/`
5. `fedot/api/main.py`
6. mirrored tests under `tests/`

## Key guarantees

- public legacy API remains intact
- TensorData paths are added as explicit entrypoints instead of hidden mode switches
- expected unsupported TensorData fit paths fail early and clearly
- bridge boundaries are deterministic and keep compatibility concerns localized
