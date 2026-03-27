---
name: fedot-pure-core-shell
description: Refactor FEDOT modules by extracting pure planning, decision, validation, normalization, filtering, or transformation logic from OOP shells while preserving public APIs, coordinator classes, and lifecycle ownership in fedot/api, fedot/core, fedot/preprocessing, fedot/extensions, fedot/remote, and mirrored tests. Use when code mixes orchestration with business rules, hides branching inside large methods, relies on stringly typed decisions, or should move toward a pure core plus thin effect shell design without breaking existing entry points.
---

# FEDOT Pure Core + Thin Effect Shell

## Overview

Use this skill to keep FEDOT's public OOP boundaries stable while moving computation-heavy rules into deterministic, testable helpers. Treat facades, builders, coordinators, repositories, and runtime adapters as shells that own lifecycle and effects; treat selection, validation, normalization, filtering, and transformation logic as pure core candidates.

## Quick Start

1. Read [repo-boundaries.md](references/repo-boundaries.md) to see which classes and packages should remain as boundary objects.
2. Read [extraction-playbook.md](references/extraction-playbook.md) to choose the extraction pattern.
3. Read [examples-from-prs.md](references/examples-from-prs.md) when you want repo-specific examples from earlier refactoring slices.
4. Read [test-obligations.md](references/test-obligations.md) before touching tests.

## Workflow

### 1. Preserve the shell

- Preserve the public constructor, facade, builder, adapter, or repository entry point unless the user explicitly asks to redesign it.
- Keep lifecycle, resource management, logging, I/O, dependency wiring, and orchestration in the shell.
- Keep compatibility with existing call sites whenever the change is presented as a refactor.

### 2. Find pure core candidates

- Extract code that derives outputs from inputs without requiring direct access to files, network, runtime state, or mutable shared objects.
- Prioritize logic for planning, validation, defaulting, recommendation, normalization, filtering, routing, and parameter resolution.
- Treat hidden branching on mode flags, presets, assumptions, and repository lookups as high-value extraction targets.

### 3. Design typed pure inputs and outputs

- Prefer small dataclasses, enums, named tuples, or narrow typed dictionaries over unstructured `dict[str, Any]` blobs when the shape is stable.
- Return values instead of mutating inbound objects whenever practical.
- Model expected failure paths explicitly. Use `Optional[...]` only when absence is normal and does not need explanation.

### 4. Rewire the shell around the pure core

- Make the shell gather raw inputs, call the pure helper, and apply side effects after the helper returns.
- Keep local mutation inside a helper only when it stays encapsulated and the external API remains deterministic.
- Avoid spreading normalization logic across multiple OOP methods after extraction.

### 5. Update tests with mirrored structure

- Add or update pure-helper unit tests near the mirrored package under `tests/`.
- Keep facade or service tests for the shell so the original entry point still has coverage.
- Add invariant-style checks when the extracted logic manipulates normalization, ordering, batching, routing, or aggregation.

## Heuristics

### Strong signals to use this skill

- A method both decides what to do and performs the action.
- A facade or builder contains long chains of conditionals for presets, assumptions, or mode selection.
- Validation and defaulting are interleaved with I/O.
- Repository or preprocessing code mixes query logic with data loading.
- A refactor needs better tests without exposing more framework internals.

### Signals to avoid over-extraction

- The code is mostly I/O or runtime control already.
- The new abstraction would only wrap one branch and add naming noise.
- The logic depends on hidden ambient state that should first be made explicit.
- The extracted helper would only proxy through to one effectful dependency.

## Output Expectations

- Keep the public API stable unless asked otherwise.
- Make the new pure collaborator easy to test in isolation.
- Leave a clear path from shell input to pure helper output.
- Mirror tests in the `tests/` tree and keep service or facade coverage when the shell remains public.

## References

- Read [repo-boundaries.md](references/repo-boundaries.md) to preserve the right OOP boundaries.
- Read [extraction-playbook.md](references/extraction-playbook.md) to choose the extraction shape.
- Read [test-obligations.md](references/test-obligations.md) to map changes into tests.
- Read [examples-from-prs.md](references/examples-from-prs.md) for FEDOT-specific examples.