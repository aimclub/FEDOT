# Modeling Patterns

## Prefer Small Named Records

Use a named record such as a `dataclass` when several values move together as one concept:

- run request
- planning result
- filter decision
- capability set
- validation result

Good signs:

- callers repeatedly pass the same group of fields together
- field names matter for readability
- tests need to assert on several related outputs

## Prefer Enums for Closed Sets

Use enums when the valid set is finite and meaningful:

- task mode
- execution mode
- backend type
- preprocessing orientation
- validation status

This is especially useful when the current code branches on strings.

## Prefer Narrow Typed Dictionaries Only at the Edge

Use `TypedDict` or raw dictionaries only when the payload is inherently dictionary-shaped because it comes from:

- JSON
- YAML
- user params
- repository metadata

Convert to named internal objects once the payload crosses the boundary.

## Prefer Value Objects Over Boolean Clusters

If a function needs several booleans that collectively describe one decision, replace them with one named result object. This often makes tests and branching much clearer.

## Current Positive Examples in the Repo

Patterns already appearing in FEDOT that are worth extending:

- `AssumptionsFilterDecision`
- `PresetSpec`
- extension contract dataclasses in `fedot/extensions/contracts.py`
- typed preprocessing decisions and rules in refactored preprocessing modules