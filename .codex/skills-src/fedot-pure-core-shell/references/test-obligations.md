# Test Obligations

## Mirror the Package Layout

Prefer tests under mirrored directories:

- `fedot/api` -> `tests/api`
- `fedot/core` -> `tests/core`
- `fedot/extensions` -> `tests/extensions`
- `fedot/preprocessing` -> `tests/preprocessing`
- `fedot/remote` -> `tests/remote`

Keep new helpers close to the mirrored package they support.

## Preserve Shell Coverage

If a public facade, builder, adapter, or repository remains in place, keep or add tests that prove:

- the public call path still works
- the shell wires dependencies correctly
- the shell uses the extracted helper result correctly

## Add Pure Helper Coverage

For extracted logic, add unit tests that cover:

- defaulting
- normalization
- filtering
- routing
- edge cases that were previously hidden in large methods

## Add Invariant Coverage When Useful

Prefer invariant-oriented tests when the helper has algebraic behavior:

- applying normalization twice gives the same result
- serialization round-trips preserve meaning
- batching or partitioning preserves total element counts
- seeded routing or sampling stays deterministic
- merging partial aggregates matches aggregate-all-at-once behavior

## Prefer Behavior Over Internals

Do not assert on incidental helper implementation details. Assert on:

- return values
- stable ordering when it matters
- visible warnings or errors
- shell behavior at the public boundary