# Implementation Flow

## Adding a New Extension

Follow this path:

1. Create an `ExtensionManifest` with at least one `ExternalModelSpec`.
2. Declare `ModelCapabilities` with supported tasks and data types.
3. Provide a callable factory with a simple signature.
4. Validate or register the manifest.
5. Smoke test the factory path.
6. Let runtime lookup and parameter rules operate from the validated spec.

## Refactoring Rules

### Keep deterministic rules pure

Prefer small deterministic helpers for:

- checking empty names
- checking duplicate models
- checking capability presence
- checking factory signature shape
- interpreting hyperparameter defaults
- mapping operations to registered extension models

### Keep effects at the boundary

Leave these at the shell:

- importing extension modules
- mutating the registry
- instantiating real model objects
- calling external runtime code

## Compatibility Guidance

When evolving the manifest contract:

- prefer additive fields over breaking replacements
- keep old authoring patterns working when possible
- add validation for newly required fields explicitly
- keep error codes stable enough for tests and callers to reason about them

## Failure Modeling Guidance

Prefer explicit contract errors for:

- invalid manifest type
- empty extension name or version
- duplicate extension name
- duplicate model name
- invalid factory
- empty task or data type capability
- factory smoke-test failure
- missing manifest in an imported module