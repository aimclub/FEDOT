# Examples From FEDOT Refactoring Slices

## Extension Contract and Runtime Bridge

Signal:

- several internal configuration points must be edited to add or interpret an extension

Extraction target:

- typed extension parameter resolution
- capability interpretation
- registry query logic

Shell to preserve:

- registration and runtime adapter entry points

Tests to expect:

- mirrored tests under `tests/extensions`
- smoke coverage for the entry path
- focused tests for typed resolution rules

## Remote Config Safety Changes

Signal:

- parsing relies on `eval`, sentinel strings, or unchecked exceptions

Extraction target:

- parse, validate, normalize, and choose-default stages

Shell to preserve:

- remote-facing API and transport integration

Tests to expect:

- mirrored tests under `tests/remote`
- success and failure cases

## API, Core, and Preprocessing Rule Extractions

Signal:

- large methods mix orchestration with assumptions, presets, defaults, or preprocessing choices

Extraction target:

- planning rules
- parameter normalization
- filter and recommendation logic

Shell to preserve:

- `Fedot` facade and related coordinators
- pipeline-facing entry points
- preprocessing orchestrators

Tests to expect:

- service or facade tests in mirrored `tests/api`
- unit tests for new helper modules
- invariants for normalization or routing when meaningful