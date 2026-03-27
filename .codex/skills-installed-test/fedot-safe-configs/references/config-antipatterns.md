# Config Antipatterns

## `eval` or Code Execution During Parsing

Do not execute config contents to obtain structure or values. Replace with explicit parsing and validation.

## Sentinel Strings for Absence

Avoid string values such as:

- `'None'`
- `'null'`
- `'default'` when the meaning is overloaded

Prefer:

- actual absence
- a real defaulting stage
- an enum for closed mode sets

## Raw `dict` Blobs Crossing Layers

Avoid passing one raw dictionary through multiple layers while each layer mutates or interprets a different subset of keys.

Prefer:

- a raw input shape at the edge
- one normalization step
- a typed internal object downstream

## Hidden Defaulting

Avoid sprinkling defaults across several methods or constructors. This makes behavior hard to reason about and harder to test.

Prefer one canonical defaulting path.

## Expected Errors Raised as Generic Exceptions

Avoid using broad exceptions for:

- missing config fields
- invalid mode names
- unsupported backend requests
- invalid parameter combinations

Represent these as structured validation outcomes whenever the failure is expected from user input.

## In-Place Mutation of Shared Config Structures

Avoid mutating inbound dictionaries that callers may continue to use. Copy or rebuild normalized data instead.

## Parsing Mixed with Side Effects

Avoid helpers that both:

- read files or repositories
- normalize values
- decide defaults
- register runtime state

Split these responsibilities.