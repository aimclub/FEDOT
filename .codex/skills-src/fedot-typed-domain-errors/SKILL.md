---
name: fedot-typed-domain-errors
description: Model FEDOT domain states, requests, results, and failures explicitly with dataclasses, enums, narrow typed records, and structured error values instead of string flags, sentinel values, raw dict blobs, or unchecked expected exceptions. Use when refactoring fedot/api, fedot/core, fedot/preprocessing, fedot/remote, or extension flows that currently hide state or domain failures inside None, loosely typed params, or branch-heavy procedural code.
---

# FEDOT Typed Domain States + Errors

## Overview

Use this skill when a FEDOT change needs clearer domain language at the type level. The goal is to replace ad hoc flags, nullable state, and hidden failure modes with small explicit domain objects that make planning, validation, and runtime decisions easier to reason about and test.

## Quick Start

1. Read [modeling-patterns.md](references/modeling-patterns.md) for the preferred shape of typed domain records.
2. Read [error-choice-guide.md](references/error-choice-guide.md) to choose between optional absence, structured domain errors, and real exceptions.
3. Read [candidate-hotspots.md](references/candidate-hotspots.md) for repo areas where this skill is likely to pay off.

## Workflow

### 1. Name the domain concept first

- Identify the real concept hidden in flags or loose dictionaries: task resolution, preset choice, capability set, routing decision, preprocessing plan, or validation failure.
- Give that concept a stable record or enum instead of passing it around as raw strings.

### 2. Close the state space when possible

- Use enums or narrow constants for modes, statuses, and backend categories when the valid set is finite.
- Replace stringly typed orchestration such as `'train'`, `'infer'`, `'export'`, or similar branch keys when callers depend on a closed set of cases.
- Make pattern matching or explicit branching consume the typed state, not the raw string.

### 3. Separate absence from failure

- Use optional values only when missing data is a normal and silent case.
- Use structured domain errors when callers need to know why a value or state is invalid.
- Do not use `None` or a sentinel string to stand for several different failure meanings at once.

### 4. Keep transport shapes at the edge

- Raw dictionaries, CLI args, JSON payloads, or repository blobs may exist at the boundary.
- Convert them once into typed internal records before deeper planning or execution logic.
- Avoid reinterpreting the same loose payload in several layers.

### 5. Protect ergonomics

- Prefer a few named records and enums over a giant type lattice.
- Introduce typed objects where they remove ambiguity or simplify tests.
- Avoid inventing ceremonial wrappers around values that do not carry domain meaning.

## Strong Signals to Use This Skill

- A function accepts or returns `dict[str, Any]` across several layers.
- Domain state is encoded in strings, booleans, or nullable fields.
- Expected failures are raised as generic exceptions or hidden behind `None`.
- Branching logic depends on preset names, task names, modes, or backend capabilities in many places.

## Output Expectations

- Name the domain concepts explicitly.
- Make expected failures testable and stable.
- Reduce hidden branching on raw strings or sentinel values.
- Keep converted typed records close to the start of the workflow.

## References

- Read [modeling-patterns.md](references/modeling-patterns.md) for type-shaping rules.
- Read [error-choice-guide.md](references/error-choice-guide.md) for failure semantics.
- Read [candidate-hotspots.md](references/candidate-hotspots.md) for repo-specific targets and examples.