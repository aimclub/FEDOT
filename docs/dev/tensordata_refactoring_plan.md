# TensorData-first refactoring and mandatory skill-routing

## Summary

Target plan file: `docs/dev/tensordata_refactoring_plan.md`.

This plan replaces the previous roadmap as the primary implementation guide.

New baseline:
- `TensorData` is the primary internal data model.
- `InputData` and the current OOP API remain a compatibility shell during the transition.
- Every new implementation slice must explicitly follow the `.codex` skill workflow.
- The next implementation work starts from `TensorData` taxonomy, creator/backend normalization, and bridge boundaries rather than from top-level API refactors.

Mandatory workflow for each next slice:
- one-sentence task classification
- one lead skill
- up to two companion skills
- mirrored tests in `tests/` in the same slice
- facade/service tests for the OOP shell
- unit and invariant tests for the pure core

## Slices

### Slice A - TensorData Stabilization
Lead skill: `fedot-safe-configs`
Companions: `fedot-pure-core-shell`, `fedot-typed-domain-errors`

Goal:
make `TensorData` and related backend/data modules internally consistent and suitable for further migration.

Changes:
- fix a canonical transition taxonomy around `tabular` and `ts`
- add legacy compatibility mapping for `table`, `multi_ts`, `text`, `image`
- extract pure rules from `TensorData` for creator resolution, backend normalization, state/task/data_type normalization, feature-target partitioning, and fallback behavior
- keep `TensorData` as an OOP data object, but turn `create(...)` and `__post_init__` into a thin shell over pure rules
- separate shell adapters for file loading, backend switching, embeddings, categorical encoding, and TS preprocessing
- remove hidden overlaps between legacy `DataTypesEnum`, backend/data tools, and preprocessing mappings

Interfaces:
- preserve `TensorData.create(...)` and `TensorData.create_lazy(...)`
- introduce typed internal spec/result/error modeling for the TensorData stack
- keep a clear compatibility mapper for legacy data types

Tests:
- mirrored `tests/core/data/...`
- creator selection
- backend selection
- taxonomy mapping
- feature/target extraction
- fallback-to-cpu behavior
- invariants: determinism, idempotence, stable row accounting

### Slice B - InputData/TensorData Bridges
Lead skill: `fedot-pure-core-shell`
Companions: `fedot-typed-domain-errors`, `fedot-invariant-tests-review`

Goal:
introduce a controlled transition between legacy and tensor-first paths.

Changes:
- add bridges: raw source -> `TensorData`, `InputData` -> `TensorData`, and `TensorData` -> `InputData` only where legacy runtime still needs it
- make `ApiDataProcessor` explicitly choose which internal path is used
- make preprocessing bridge-aware for legacy `InputData` and tensor-aware `TensorData`
- preserve the public `Fedot.fit/predict/...` API

Tests:
- mirrored `tests/api/...`, `tests/preprocessing/...`, `tests/core/data/...`
- facade compatibility tests
- bridge correctness tests
- invariants: round-trip where promised, deterministic conversion, no hidden mutation leaks

### Slice C - Tensor-aware Pipeline and Runtime Integration
Lead skill: `fedot-pure-core-shell`
Companions: `fedot-invariant-tests-review`, `fedot-extension-contract`

Goal:
move selected internal execution paths to `TensorData` without a full pipeline rewrite.

Changes:
- choose a limited set of pipeline/runtime paths that consume `TensorData` directly
- keep `Pipeline` OOP-first
- add tensor-aware preprocessing and postprocessing plans where useful
- verify extension/runtime compatibility with tensor/backend-aware params

Tests:
- mirrored `tests/core/pipelines/...`
- service-level runtime compatibility tests
- invariants for preprocessing and postprocessing boundaries

### Slice D - Industrial Convergence
Lead skill: `fedot-pure-core-shell`
Companions: `fedot-typed-domain-errors`, `fedot-invariant-tests-review`

Goal:
converge `fedot/industrial` only after the core TensorData path is stable.

Changes:
- reuse the stabilized TensorData bridges
- avoid a competing tensor architecture in industrial modules
- converge only through shared backend/data contracts

## Test strategy updates

- canonical path for new tests is `tests/`
- legacy `test/` stays as a temporary compatibility layer
- TensorData-first slices must add mirrored tests in `tests/core/data/...`, `tests/preprocessing/...`, and `tests/api/...` when relevant
- marker policy stays `unit`, `integration`, `property`, `slow`
- GPU, UCR, and text-embedding-heavy scenarios must stay isolated with markers

## Assumptions and defaults

- `TensorData` is the primary internal model.
- `InputData` remains a compatibility shell for near-term slices.
- The next implementation slice is Slice A.
- Until Slice A is stabilized, do not resume broad API-layer refactors except at TensorData integration points.
- `.codex` skills are mandatory process constraints for future slices.

## Immediate implementation order

1. save this plan in `docs/dev/tensordata_refactoring_plan.md`
2. identify TensorData taxonomy and normalization hotspots
3. extract pure rules from `TensorData.create(...)` and `TensorData.__post_init__`
4. stabilize backend, data_type, and state compatibility mapping
5. add mirrored `tests/core/data/...`
6. prepare the first commit-sized vertical slice around TensorData normalization rules
