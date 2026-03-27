# Skill Catalog

## `$fedot-pure-core-shell`

Use for:

- extracting planning, validation, filtering, and normalization rules from OOP shells
- preserving public facades while simplifying internal logic
- splitting shell effects from deterministic helpers

Typical packages:

- `fedot/api`
- `fedot/core`
- `fedot/preprocessing`
- `fedot/remote`

## `$fedot-invariant-tests-review`

Use for:

- choosing the right test layer after refactors
- PR review against the FEDOT FP checklist
- adding invariants for normalization, routing, batching, merging, and serialization

Typical packages:

- `tests/*`
- any mirrored refactor touching public boundaries and pure helpers

## `$fedot-safe-configs`

Use for:

- replacing `eval`
- removing sentinel strings and hidden null semantics
- redesigning parse, validate, normalize, and default flows

Typical packages:

- `fedot/remote`
- loader or parser-heavy API flows
- manifest or metadata parsing paths

## `$fedot-extension-contract`

Use for:

- external model manifest design
- registry flow
- smoke testing factories
- runtime adapter behavior downstream of validated manifests

Typical packages:

- `fedot/extensions`
- extension-related runtime integration

## `$fedot-typed-domain-errors`

Use for:

- replacing stringly typed orchestration and nullable state
- introducing named domain records, enums, and structured failures
- clarifying requests, results, plans, and error surfaces

Typical packages:

- `fedot/api`
- `fedot/core`
- `fedot/preprocessing`
- `fedot/remote`
- `fedot/extensions`

## `$fedot-refactor-router`

Use for:

- ambiguous tasks
- mixed refactors that span several concerns
- early planning before implementation

Typical packages:

- any combination of the above