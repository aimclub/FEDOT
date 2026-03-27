---
name: fedot-refactor-router
description: Classify FEDOT changes and route work to specialized skills such as pure core extraction, invariant tests review, safe configs, extension contract, or typed domain errors. Use when a request spans several architectural concerns, when the right skill is unclear, or when planning a refactor or PR before implementation.
---

# FEDOT Refactor Router

## Overview

Use this skill when you need to decide which FEDOT skill should lead the work. The router does not replace specialized skills; it chooses the primary one, identifies secondary companions, and gives a short implementation order.

## Quick Start

1. Read [routing-matrix.md](references/routing-matrix.md) to map the request to a primary skill.
2. Read [package-hotspots.md](references/package-hotspots.md) to use the touched packages as routing clues.
3. Read [example-prompts.md](references/example-prompts.md) when the request is ambiguous and you want similar examples.

## Routing Workflow

### 1. Identify the dominant change shape

Classify the request by its main outcome:

- extracting pure decision or planning logic
- clarifying typed states or domain errors
- making config parsing safe
- adding or refactoring extension integration
- deciding what tests or review criteria are needed

### 2. Choose one primary skill

Pick the skill that owns the highest-risk or most central concern. Avoid invoking three skills equally unless the request truly spans them.

### 3. Add at most two secondary skills

Use secondary skills only when they materially shape the solution, for example:

- `fedot-pure-core-shell` + `fedot-invariant-tests-review`
- `fedot-safe-configs` + `fedot-typed-domain-errors`
- `fedot-extension-contract` + `fedot-invariant-tests-review`

### 4. Sequence the work

Default order:

1. choose the domain model and safe parsing strategy if needed
2. extract or reshape the pure logic
3. update integration or shell code
4. add tests and review against invariants

## Routing Rules

### Route to `$fedot-pure-core-shell` when

- the main problem is mixed orchestration and rule logic
- large methods hide branching, normalization, or planning
- the public API should stay intact while internals become cleaner

### Route to `$fedot-invariant-tests-review` when

- the user asks for review
- the change is mostly about missing tests, regressions, or validation of a refactor
- you need to choose between facade tests, helper tests, and invariant checks

### Route to `$fedot-safe-configs` when

- parsing uses `eval`, sentinels, raw dicts, or unchecked expected exceptions
- the main change is parse, validate, normalize, and default behavior

### Route to `$fedot-extension-contract` when

- the request touches `fedot/extensions`
- new external models or manifests are involved
- registry, discovery, smoke-test, or runtime adapter flow is central

### Route to `$fedot-typed-domain-errors` when

- states, results, or failures are implicit
- strings, booleans, or nullable fields stand in for domain concepts
- the main win is clearer typed modeling rather than shell extraction alone

## Output Expectations

Produce a short routing answer with:

- the primary skill
- up to two secondary skills
- one sentence on why
- a suggested implementation order

## References

- Read [routing-matrix.md](references/routing-matrix.md) for the main decision table.
- Read [package-hotspots.md](references/package-hotspots.md) for package-based cues.
- Read [example-prompts.md](references/example-prompts.md) for concrete routing examples.