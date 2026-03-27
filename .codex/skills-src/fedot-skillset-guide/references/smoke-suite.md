# Smoke Suite

## Purpose

Use these scenarios after changing the skillset to confirm that routing and composition still feel correct.

## Scenario 1: Extract typed planner from facade

Prompt:

- "Extract typed planning rules from a `Fedot` facade method while preserving the public API and mirrored tests."

Expected lead skill:

- `$fedot-pure-core-shell`

Expected companion:

- `$fedot-invariant-tests-review`

## Scenario 2: Remove `eval` and model failures explicitly

Prompt:

- "Replace unsafe remote config parsing with typed validation and explicit failure results."

Expected lead skill:

- `$fedot-safe-configs`

Expected companions:

- `$fedot-typed-domain-errors`
- `$fedot-invariant-tests-review`

## Scenario 3: Add an external model manifest

Prompt:

- "Integrate a new external model through the extension manifest, registry, and smoke-test flow."

Expected lead skill:

- `$fedot-extension-contract`

Expected companion:

- `$fedot-invariant-tests-review`

## Scenario 4: Clarify ambiguous routing

Prompt:

- "This PR touches `fedot/extensions`, typed failures, and tests. Which skill should lead?"

Expected lead skill:

- `$fedot-refactor-router`

## Scenario 5: Replace stringly state and nullable result objects

Prompt:

- "Refactor task and backend resolution so states and failures are explicit domain objects."

Expected lead skill:

- `$fedot-typed-domain-errors`

Expected companions:

- `$fedot-pure-core-shell`
- `$fedot-invariant-tests-review`