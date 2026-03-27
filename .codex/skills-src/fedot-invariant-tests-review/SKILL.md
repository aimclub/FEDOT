---
name: fedot-invariant-tests-review
description: Write or review FEDOT tests for FP-informed refactors, including facade or service tests for OOP coordinators, unit tests for pure collaborators, and invariant-oriented checks such as determinism, round trips, idempotence, monotonicity, boundary preservation, and aggregation laws. Use when changes touch samplers, schedulers, config merges, repository queries, preprocessing or pipeline transforms, parameter normalization, extension contracts, or when reviewing PRs against the FEDOT FP checklist.
---

# FEDOT Invariant Tests + Review

## Overview

Use this skill to decide what kind of test FEDOT code needs after an FP-informed refactor and to review PRs with the repository's architectural checklist in mind. Favor tests that lock down behavior and invariants, not incidental implementation details.

## Quick Start

1. Read [test-placement-map.md](references/test-placement-map.md) to decide where tests belong.
2. Read [invariant-catalog.md](references/invariant-catalog.md) to choose behavior worth locking down.
3. Read [review-checklist.md](references/review-checklist.md) when asked to review a PR or evaluate architectural risk.

## Workflow

### 1. Classify the target

- If the code is a facade, builder, adapter, or service boundary, plan boundary tests.
- If the code is a pure helper, planner, normalizer, parser, or selector, plan direct unit tests.
- If the code combines several data transformations, look for invariants in addition to examples.

### 2. Cover the main regression path

- Preserve at least one test through the public path when the entry point remains public.
- Add focused tests for the extracted pure logic instead of only increasing shell mocks.
- Cover the failure mode that motivated the refactor if one is known.

### 3. Add invariants where behavior should hold broadly

- Favor invariants for normalization, parsing, batching, routing, sampling, merges, and serialization.
- Use parametrization or small generated cases even if no property-testing library is available.
- Keep assertions on behavior, not private helper internals.

### 4. Review PRs with findings first

- If asked for review, list bugs, risks, missing tests, or behavioral regressions before summaries.
- Use the FP checklist to decide whether the code hides rules, errors, or unsafe mutation.
- Call out missing facade coverage and missing pure-helper coverage separately.

## Test Selection Rules

### Prefer facade or service tests when

- the public entry point remains supported
- orchestration order matters
- wiring to collaborators is part of the contract
- the refactor claims backward compatibility

### Prefer direct unit tests when

- a helper now computes plans, defaults, filters, or routing decisions
- the function has explicit inputs and outputs
- a previous large method was split into deterministic stages

### Prefer invariant-style tests when

- the logic should remain deterministic under a seed
- normalization should be idempotent
- serialization should round-trip
- partitioning or batching should conserve counts
- aggregate merge order should not change meaning

## Review Expectations

- Name the missing or weak test type, not just "tests are missing".
- Explain why the current coverage could miss a regression.
- Tie review comments back to public behavior, invariants, or architectural rules.

## References

- Read [test-placement-map.md](references/test-placement-map.md) for mirrored test placement.
- Read [invariant-catalog.md](references/invariant-catalog.md) for common FEDOT invariants.
- Read [review-checklist.md](references/review-checklist.md) for PR review questions.