# Sampling Stage Integration Review
Date: 2026-03-06
Scope: changes for `sampling_config` + pre-fit `sampling_stage` integration into `Fedot.fit`.

## Final Stage Status
- Attempted final test run:
  - `python -m pytest ...` -> `No module named pytest`
- Final pytest stage is skipped as not runnable in current environment.
- Fallback verification completed:
  - `python -m py_compile` passed for all changed/new Python files.

## Change Scope (Implemented)
- API/defaults: `sampling_config` added and validated.
- New subsystem: `fedot/api/sampling_stage/{config.py,providers.py,executor.py}`.
- Fit integration: sampling stage executed before composition, metadata exposed.
- Optional dependency: `fedot[sampling_zoo]` extra added.
- Tests: unit + integration tests for config, provider, executor, and fit behavior.
- Docs/examples: advanced guide, README section, classification example note.

## 1) Architecture Review

### A1. Provider Contract Is Heuristic and Version-Sensitive
Problem:
- `SamplingZooProvider` discovers indices through multiple fallback paths (`sample_indices`, attrs, `get_partitions`) without a strict external contract.

Why it matters:
- Changes in Sampling Zoo internals can break extraction logic, causing hard failures or incorrect sampling behavior.

Options:
1. Do nothing.
- Effort: Low
- Risk: Medium
- Payoff: Low
- Maintenance cost: Medium

2. Define and enforce a strict provider adapter contract for FEDOT integration.
- Effort: Medium
- Risk: Low
- Payoff: High
- Maintenance cost: Low

3. Maintain versioned adapters (`sampling_zoo_v1`, `sampling_zoo_v2`) with explicit compatibility checks.
- Effort: Medium/High
- Risk: Low
- Payoff: High
- Maintenance cost: Medium

Recommended:
- Option 2 for near term; move to Option 3 when multiple Sampling Zoo API generations must be supported.

### A2. Subset Construction May Share Mutable Supplementary State
Problem:
- `SamplingStageExecutor._subset_by_positions` reuses `data.supplementary_data` reference when creating reduced `InputData`.

Why it matters:
- Mutable shared state can create non-obvious side effects across stages/pipelines.

Options:
1. Do nothing.
- Effort: Low
- Risk: Medium
- Payoff: Low
- Maintenance cost: Medium

2. Deep-copy `supplementary_data` for sampled dataset.
- Effort: Low
- Risk: Low
- Payoff: Medium
- Maintenance cost: Low

3. Introduce immutable or copy-on-write semantics for supplementary metadata.
- Effort: Medium/High
- Risk: Low
- Payoff: High
- Maintenance cost: Medium

Recommended:
- Option 2 now; Option 3 only if broader data mutability problems appear.

## 2) Code Quality Review

### C1. Guard Validation Is Partially Type-Specific
Problem:
- Heavy-parameter guards in `validate_sampling_config` primarily check integer forms (`n_partitions`, `sample_size`, etc.), while some non-int shapes may bypass limits.

Why it matters:
- Invalid or heavy configs may slip through validation and produce expensive runtime behavior.

Options:
1. Do nothing.
- Effort: Low
- Risk: Medium
- Payoff: Low
- Maintenance cost: Medium

2. Normalize and validate all accepted numeric representations (int/float/list/tuple where relevant).
- Effort: Medium
- Risk: Low
- Payoff: High
- Maintenance cost: Low

3. Add provider-specific schema validation plugins.
- Effort: Medium/High
- Risk: Low
- Payoff: High
- Maintenance cost: Medium

Recommended:
- Option 2 in V1 hardening; Option 3 only when multiple providers are active.

### C2. Final Sampling Randomly Re-Selects from Extracted Indices
Problem:
- After extracting candidate indices from strategy output, provider may randomly choose a subset up to `sample_size`.

Why it matters:
- If strategy output is already ranked/structured, extra random reduction can weaken algorithm intent and reproducibility semantics.

Options:
1. Do nothing.
- Effort: Low
- Risk: Medium
- Payoff: Low
- Maintenance cost: Low

2. Prefer strategy-native final selection when available; fallback to random only if needed.
- Effort: Medium
- Risk: Low
- Payoff: High
- Maintenance cost: Low

3. Require strategy to return exactly final indices count and fail otherwise.
- Effort: Medium
- Risk: Medium
- Payoff: High
- Maintenance cost: Medium

Recommended:
- Option 2 for compatibility + quality balance.

## 3) Test Review

### T1. No Executed End-to-End Test with Real Sampling Zoo in This Environment
Problem:
- Tests were authored, but final execution is blocked by missing `pytest` package; no runtime E2E signal with installed optional dependency.

Why it matters:
- Integration defects can remain hidden until real environment execution.

Options:
1. Do nothing.
- Effort: Low
- Risk: High
- Payoff: Low
- Maintenance cost: Low

2. Add CI lane with `fedot[sampling_zoo]` and run dedicated markers.
- Effort: Medium
- Risk: Low
- Payoff: High
- Maintenance cost: Medium

3. Add nightly AMLB-style smoke benchmark for sampling stage.
- Effort: Medium/High
- Risk: Low
- Payoff: High
- Maintenance cost: Medium/High

Recommended:
- Option 2 immediately; Option 3 as performance/quality observability extension.

### T2. Missing Regression Cases for DataFrame Features and Metadata Isolation
Problem:
- Tests mostly use numpy-like datasets and mocked provider paths.

Why it matters:
- Potential regressions in DataFrame handling and shared supplementary metadata may not be detected.

Options:
1. Do nothing.
- Effort: Low
- Risk: Medium
- Payoff: Low
- Maintenance cost: Low

2. Add unit/integration tests for DataFrame features, categorical columns, and supplementary metadata isolation.
- Effort: Low/Medium
- Risk: Low
- Payoff: High
- Maintenance cost: Low

3. Add property-based tests for sampling indices and data consistency invariants.
- Effort: Medium
- Risk: Low
- Payoff: High
- Maintenance cost: Medium

Recommended:
- Option 2 now; Option 3 later if index-related bugs appear in production.

## 4) Performance Review

### P1. Repeated Feature Encoding for Each Candidate Ratio
Problem:
- The effective-size protocol rebuilds training matrices and model fits for each candidate.

Why it matters:
- Sampling overhead can consume a meaningful part of budget on medium/large tabular datasets.

Options:
1. Do nothing.
- Effort: Low
- Risk: Medium
- Payoff: Low
- Maintenance cost: Low

2. Cache transformed validation matrix and reusable feature engineering outputs.
- Effort: Medium
- Risk: Low
- Payoff: Medium/High
- Maintenance cost: Medium

3. Add adaptive candidate schedule with early elimination and dynamic stopping.
- Effort: Medium
- Risk: Low
- Payoff: High
- Maintenance cost: Medium

Recommended:
- Option 3 plus targeted caching from Option 2 for the largest workloads.

### P2. Fixed RF Baseline Complexity (`n_estimators=100`)
Problem:
- Baseline model cost is fixed and may be too expensive under tight time budgets.

Why it matters:
- High stage cost can reduce AutoML search time and offset sampling benefit.

Options:
1. Do nothing.
- Effort: Low
- Risk: Medium
- Payoff: Low
- Maintenance cost: Low

2. Add lightweight baseline config (`n_estimators`, depth, model family) in `sampling_config`.
- Effort: Medium
- Risk: Low
- Payoff: High
- Maintenance cost: Low

3. Auto-scale baseline complexity from dataset size and stage budget.
- Effort: Medium/High
- Risk: Medium
- Payoff: High
- Maintenance cost: Medium

Recommended:
- Option 2 first; Option 3 later when benchmark telemetry is available.

## Consolidated Recommendation
- The current implementation is a solid V1 integration aligned with fail-fast and dynamic cap constraints.
- Main hardening targets before production broad rollout:
  - enforce stricter provider contract,
  - isolate mutable dataset metadata,
  - extend guard validation,
  - execute CI with real optional dependency.