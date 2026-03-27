# Routing Matrix

## Primary Skill by Change Shape

### Pure refactor with preserved API

Primary skill:

- `$fedot-pure-core-shell`

Secondary skill:

- `$fedot-invariant-tests-review`

Use when:

- the request says extract rules, split shell and core, or keep facade and clean internals

### Safer parsing or validation flow

Primary skill:

- `$fedot-safe-configs`

Secondary skills:

- `$fedot-typed-domain-errors`
- `$fedot-invariant-tests-review`

Use when:

- the request mentions config parsing, `eval`, sentinel values, normalization, defaults, or structured validation

### Explicit state, result, or failure modeling

Primary skill:

- `$fedot-typed-domain-errors`

Secondary skills:

- `$fedot-pure-core-shell`
- `$fedot-invariant-tests-review`

Use when:

- the main issue is stringly typed orchestration, ambiguous `None`, or weak domain vocabulary

### External model integration

Primary skill:

- `$fedot-extension-contract`

Secondary skills:

- `$fedot-invariant-tests-review`
- `$fedot-typed-domain-errors`

Use when:

- manifests, registry flow, runtime adapters, extension discovery, or smoke tests are central

### Review-focused request

Primary skill:

- `$fedot-invariant-tests-review`

Secondary skills:

- whichever specialized implementation skill matches the touched area

Use when:

- the user asks to review a PR, judge refactor quality, or identify risks and missing tests