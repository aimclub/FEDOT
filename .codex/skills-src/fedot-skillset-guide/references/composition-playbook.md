# Composition Playbook

## Common Pairings

### `$fedot-pure-core-shell` + `$fedot-invariant-tests-review`

Use when:

- logic is extracted from a facade or coordinator
- the public path must stay stable
- regression and helper-level tests both matter

### `$fedot-safe-configs` + `$fedot-typed-domain-errors`

Use when:

- config parsing is unsafe
- the parse result and failures are also too implicit
- you need both safer flow and clearer domain language

### `$fedot-extension-contract` + `$fedot-invariant-tests-review`

Use when:

- registry or runtime behavior changes
- manifest validation and smoke tests must stay trustworthy

### `$fedot-typed-domain-errors` + `$fedot-pure-core-shell`

Use when:

- shell extraction alone is not enough because domain vocabulary is still too loose
- the refactor should produce explicit plans, decisions, or errors

## Default Sequencing Rules

1. If routing is unclear, start with `$fedot-refactor-router`.
2. If the task changes data interpretation, choose typed modeling and safe config strategy early.
3. If the task changes orchestration internals, do pure-core extraction before test polishing.
4. If the task changes behavior visibility, finish with `$fedot-invariant-tests-review`.

## Overlap to Avoid

Avoid using several skills for the same sentence of advice. Examples:

- let `$fedot-safe-configs` own parsing stages
- let `$fedot-typed-domain-errors` own semantic state and failure shapes
- let `$fedot-pure-core-shell` own shell-versus-core decomposition
- let `$fedot-invariant-tests-review` own coverage and review judgments