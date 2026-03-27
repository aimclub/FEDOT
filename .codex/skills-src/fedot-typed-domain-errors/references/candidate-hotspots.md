# Candidate Hotspots

## `fedot/api`

Good candidates include:

- task resolution
- run planning
- preset parsing and filtering
- recommendation rules
- assumptions builders

Examples already moving in the right direction:

- `PresetSpec` in `fedot/api/api_utils/assumptions/assumption_rules.py`
- typed planner and service tests under `tests/api/api_utils`

## `fedot/preprocessing`

Good candidates include:

- source selection
- merge plans
- optional preprocessing decisions
- orientation or encoding strategy choices

Use named decision objects when several related booleans or source names travel together.

## `fedot/remote`

Good candidates include:

- remote task mode
- config parsing
- dataset and execution setup
- backend or client selection

This area benefits from separating parse failures, absent values, and environment failures.

## `fedot/extensions`

Good candidates include:

- manifest errors
- capabilities
- external model specs
- parameter resolution results

The extension layer already contains explicit dataclasses and error records. Prefer extending that style instead of falling back to raw metadata handling.

## Review Clues

A hotspot likely needs this skill when:

- several functions pass the same raw dictionary onward
- branch conditions inspect many string keys
- `None` values accumulate and later require interpretation
- the code returns tuples whose meaning is hard to remember without reading the implementation