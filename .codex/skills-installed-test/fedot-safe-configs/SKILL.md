---
name: fedot-safe-configs
description: Implement safe typed configuration parsing, validation, defaulting, and normalization in FEDOT without eval, sentinel strings such as 'None', hidden null semantics, or unchecked expected exceptions. Use when working on fedot/remote, parameter loaders, repository metadata, extension manifests, checkpoint or backend selection, or any parser or validator that currently relies on raw dict blobs, string mode flags, or mixed parse plus effect logic.
---

# FEDOT Safe Configs

## Overview

Use this skill to turn fragile config and parameter handling into explicit parse, validate, normalize, and default flows. Keep loading and I/O at the boundary; make the data interpretation rules deterministic and typed enough to review and test directly.

## Quick Start

1. Read [config-antipatterns.md](references/config-antipatterns.md) to spot unsafe patterns quickly.
2. Read [safe-parsing-rules.md](references/safe-parsing-rules.md) to structure the replacement flow.
3. Read [error-modeling-guide.md](references/error-modeling-guide.md) to choose between optional absence, structured validation errors, and true exceptions.

## Workflow

### 1. Freeze the external input shape

- Identify the raw payload source: function params, JSON, YAML, repository metadata, CLI args, or extension manifest input.
- Separate raw transport shape from the internal typed shape you want downstream code to use.

### 2. Split the flow into stages

- Parse raw input into a minimal raw structure.
- Validate required fields and combinations.
- Normalize values into canonical forms.
- Apply explicit defaults.
- Return a typed result or structured failure.

### 3. Replace unsafe patterns

- Remove `eval` and other code-executing parse paths.
- Remove sentinel strings such as `'None'` when absence or a typed enum would be clearer.
- Replace expected-exception control flow with structured validation results.
- Replace stringly typed mode flags with enums or named constants where the state space is closed.

### 4. Keep the shell thin

- Let the shell load raw config text, call the parser or normalizer, and react to the typed result.
- Do not spread defaulting and validation across several callers.
- Keep side effects and repository access outside the pure parse and normalize stages.

### 5. Lock behavior with tests

- Add success cases for representative valid inputs.
- Add failure cases for missing fields, invalid combinations, and unsupported modes.
- Add idempotence or round-trip checks when normalization or serialization is involved.

## Modeling Guidance

### Prefer explicit types for stable shapes

- Use `dataclass` for stable config records.
- Use `Enum` for closed mode or backend sets.
- Use `TypedDict` for raw payloads only when the payload really is dictionary-shaped.

### Prefer optional absence only when silence is acceptable

- Use `Optional[T]` when "value not provided" is a normal state.
- Use structured validation errors when the caller should know why the input is bad.
- Reserve exceptions for unexpected programmer errors or environment failures.

### Avoid over-typing short-lived glue

- Do not invent a large type hierarchy for one local parser.
- Keep the number of stages small and explicit.
- Prefer one canonical normalization path over many wrappers.

## References

- Read [config-antipatterns.md](references/config-antipatterns.md) to identify hazards.
- Read [safe-parsing-rules.md](references/safe-parsing-rules.md) to design the replacement pipeline.
- Read [error-modeling-guide.md](references/error-modeling-guide.md) to choose error semantics.