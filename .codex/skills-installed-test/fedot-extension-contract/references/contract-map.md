# Contract Map

## Canonical Entities

The current extension contract already exists in `fedot/extensions/contracts.py`.

Use these entities as the stable vocabulary:

- `ModelHyperparamsSchema`
- `ModelCapabilities`
- `ExternalModelSpec`
- `ExtensionManifest`
- `ExtensionError`
- `RegisteredExtension`

## Current Responsibility Split

### `fedot/extensions/contracts.py`

Defines the typed data contract for manifests, model specs, capabilities, hyperparameter schema, and contract-level errors.

### `fedot/extensions/registry.py`

Owns:

- manifest validation
- external model spec validation
- extension registration
- extension discovery from modules
- smoke testing of factories
- in-memory registry state

### `fedot/extensions/parameter_rules.py`

Owns typed extension parameter resolution after the model spec is known.

### `fedot/extensions/operation_rules.py`

Owns operation discovery or query rules based on registered extensions.

### `fedot/extensions/runtime_rules.py`

Owns runtime-facing lookup and model instantiation rules based on validated specs.

### `fedot/core/operations/extension_model.py`

Acts as a runtime adapter over manifest-registered external models.

## Canonical Manifest Flow

1. Author `FEDOT_EXTENSION_MANIFEST`.
2. Load or validate the manifest.
3. Register it in the extension registry.
4. Smoke test factories.
5. Resolve parameters and runtime behavior from the registered spec.

## Practical Principle

The more work that can happen from a validated `ExternalModelSpec`, the less downstream code needs to care about raw extension metadata.