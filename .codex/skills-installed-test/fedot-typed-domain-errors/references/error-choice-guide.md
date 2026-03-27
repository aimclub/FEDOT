# Error Choice Guide

## Use Optional Absence When

- a value may genuinely be absent
- the caller does not need an explanation
- the missing value is not itself a domain failure

Examples:

- an optional modifier suffix
- a not-yet-computed auxiliary field
- an optional source override

## Use Structured Domain Errors When

- input may be invalid in expected ways
- the caller should know why planning or validation failed
- tests should assert on failure categories
- the failure belongs to domain logic rather than runtime infrastructure

Examples:

- unsupported task or backend
- invalid preset combination
- malformed manifest
- incompatible config options

## Use Exceptions When

- the environment failed
- an imported module cannot be loaded
- a dependency is unavailable
- the code reaches an unexpected programmer error state

## Anti-Patterns

Avoid:

- using one sentinel such as `'None'` or an empty string to represent many meanings
- returning `None` for both “not found” and “invalid input”
- throwing generic exceptions for ordinary validation failures
- burying the real reason for failure in log text only

## Practical Rule

If a caller would immediately ask "why did this fail?", use a structured domain error instead of optional absence.