# Invariant Catalog

## Determinism

Use when logic depends on:

- seeds
- ordered candidate selection
- repeated planning from the same inputs

Typical assertion:

- the same input and seed produce the same output

## Idempotence

Use when code normalizes or canonicalizes values.

Typical assertions:

- `normalize(normalize(x)) == normalize(x)`
- repeated parameter cleanup does not drift

## Round Trip

Use when code serializes, deserializes, or converts representations.

Typical assertions:

- `decode(encode(x)) == x`
- template export and import preserve meaning

## Conservation

Use when code partitions, batches, shards, merges, or samples.

Typical assertions:

- the total item count is preserved
- partition outputs stay within bounds
- no element appears outside allowed partitions

## Monotonicity

Use when a stronger filter, threshold, or policy should only narrow outcomes.

Typical assertions:

- increasing strictness never increases the accepted set

## Associative Aggregation

Use when combining metrics, warnings, statistics, or partial results.

Typical assertions:

- merge order does not change the final aggregate
- aggregating partial chunks matches aggregating the whole input

## Explicit Failure Surface

Use when a parser, validator, or selector should fail predictably.

Typical assertions:

- invalid input returns a structured failure
- expected bad input does not leak unchecked exceptions

## Practical Note

If no property-testing library is in play, approximate invariants with:

- parametrized tests
- small case tables
- seed loops
- carefully chosen boundary examples