# FEDOT Skills Smoke Prompts

Use these prompts to do a quick forward pass over the skill set. Each prompt should be solvable mainly from the skill plus repo context.

## `$fedot-pure-core-shell`

1. Extract planning rules from a FEDOT facade method into a pure helper while preserving the public API and mirrored tests.
2. Refactor a preprocessing orchestrator that mixes source selection, defaulting, and execution into a thin shell over deterministic rule functions.
3. Split repository query logic from I/O in `fedot/core` without changing the caller-facing boundary.

## `$fedot-invariant-tests-review`

1. Review a refactor that extracted normalization logic from `fedot/api` and identify missing facade tests, helper tests, and invariants.
2. Add tests for a deterministic planner and explain which assertions belong to public behavior versus helper internals.
3. Design invariant-oriented tests for normalization, batching, or merge behavior in a FEDOT module.

## `$fedot-safe-configs`

1. Replace an `eval`-based config parser in `fedot/remote` with explicit parse, validate, normalize, and default stages.
2. Remove sentinel `'None'` handling from a FEDOT parameter flow and replace it with typed absence or structured validation errors.
3. Consolidate scattered defaulting of config values into one canonical normalization path.

## `$fedot-extension-contract`

1. Add a new external model through `ExtensionManifest` and keep the registration, smoke-test, and runtime path canonical.
2. Refactor extension registry logic so runtime behavior depends only on validated manifest data.
3. Review a PR that changes `fedot/extensions/registry.py` and identify risks in duplicate detection, smoke testing, or manifest evolution.

## `$fedot-typed-domain-errors`

1. Replace string flags and nullable state in a FEDOT planner with explicit typed records and enums.
2. Model backend selection failures as structured domain errors rather than generic exceptions.
3. Convert a raw dict-based result flowing through several functions into a named domain object with explicit semantics.

## `$fedot-refactor-router`

1. Classify a request that both removes `eval` from remote config parsing and adds better failure modeling, then choose the primary and secondary skills.
2. Route a PR that extracts pure rules from `Fedot` and needs regression coverage.
3. Decide which skill should lead when a change touches `fedot/extensions`, typed errors, and tests in one branch.

## Composite Smoke Prompts

1. Remove unsafe remote config parsing, make failure modes explicit, and add regression tests.
   Expected lead: `$fedot-safe-configs`

2. Extract pure planning logic from `fedot/api` and preserve public behavior with mirrored tests.
   Expected lead: `$fedot-pure-core-shell`

3. Refactor external model registration around explicit manifests and validate the runtime path.
   Expected lead: `$fedot-extension-contract`