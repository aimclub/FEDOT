# Safe Parsing Rules

## Canonical Pipeline

Use this stage order whenever possible:

1. Load raw bytes or text at the shell.
2. Parse into a raw Python structure.
3. Validate presence, shape, and cross-field rules.
4. Normalize values into canonical forms.
5. Apply defaults explicitly.
6. Return a typed result or a structured failure.

Keep stages 3 through 5 deterministic and free of side effects.

## Python-Oriented Implementation Patterns

Prefer:

- `dataclass` for internal stable config objects
- `Enum` for modes, backends, or closed capabilities
- small helper functions for each stage
- one top-level pure `parse_*` or `normalize_*` function that coordinates the stages

Use `TypedDict` only for edge payloads that truly remain dictionary-shaped.

## Validation Guidance

Validate:

- required fields
- mutually exclusive options
- dependent fields
- supported backend or mode values
- shape or type mismatches

Decide early whether to fail fast or collect multiple validation errors. Be consistent inside one parser.

## Normalization Guidance

Normalize:

- aliases to canonical names
- string flags to enums or constants
- paths or identifiers into one accepted representation
- defaults in one place only

Normalization should be deterministic and usually idempotent.

## Shell Boundary Guidance

Let the shell:

- obtain the raw payload
- call the pure parser or normalizer
- handle files, logging, and transport concerns
- convert structured failures into user-facing messages if needed

Do not let the pure parser fetch data from external systems.