# Error Modeling Guide

## Choose the Failure Shape Deliberately

Use the lightest shape that still communicates the needed meaning.

## `Optional[T]`

Use when:

- absence is a normal case
- the caller does not need an explanation
- a missing value is not an error by itself

Do not use `Optional[T]` to hide malformed input.

## Structured Validation Result

Use when:

- failure is expected from user or config input
- the caller should know why validation failed
- you want tests to assert on explicit failure categories

Common shapes:

- a small error dataclass
- an enum plus message
- a result object carrying either value or errors

## Exceptions

Use exceptions for:

- programmer mistakes
- truly unexpected states
- environment failures such as I/O, unavailable dependencies, or corrupted external resources

Do not use generic exceptions as the main mechanism for expected config validation.

## Useful Error Categories

Model categories such as:

- missing field
- invalid value
- unsupported backend
- incompatible option combination
- malformed structure

Keep categories stable enough that tests and callers can reason about them.