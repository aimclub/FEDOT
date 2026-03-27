---
name: fedot-extension-contract
description: Design, validate, and refactor FEDOT external model integration through ExtensionManifest, ExternalModelSpec, ModelCapabilities, registry logic, parameter rules, operation discovery, and runtime adapters. Use when adding new external models, changing manifest shape, updating fedot/extensions, refactoring extension registration or smoke-test flow, or reviewing PRs that touch manifest-driven runtime integration.
---

# FEDOT Extension Contract

## Overview

Use this skill when the change concerns how FEDOT discovers, validates, registers, and executes external models. Keep the extension surface explicit and typed so that new integrations follow one canonical flow: create manifest, validate or register, smoke test, then expose runtime behavior.

## Quick Start

1. Read [contract-map.md](references/contract-map.md) to anchor the canonical entities and current files.
2. Read [implementation-flow.md](references/implementation-flow.md) to preserve the extension lifecycle.
3. Read [test-and-review.md](references/test-and-review.md) before changing registry logic, parameter resolution, or runtime behavior.

## Workflow

### 1. Preserve the canonical contract

- Use `ExtensionManifest` as the entry contract for one extension package.
- Use `ExternalModelSpec` for each model exposed by the extension.
- Use `ModelCapabilities` and `ModelHyperparamsSchema` to describe supported tasks, data types, tags, and parameter expectations.
- Use `ExtensionError` for expected contract failures.

### 2. Keep one integration path

- The default path should remain `create manifest -> validate/register -> smoke test -> runtime use`.
- Avoid introducing alternative side channels that bypass validation.
- Keep manifest discovery separate from runtime instantiation.

### 3. Isolate pure rule layers

- Keep manifest validation, parameter resolution, and operation discovery in deterministic helpers.
- Keep module imports, registry mutation, and estimator instantiation at the shell boundary.
- Make runtime adapter code depend on validated specs rather than raw dictionaries.

### 4. Model failures explicitly

- Return structured contract errors for invalid manifests, duplicate models, unsupported signatures, or failed smoke tests.
- Keep duplicate detection, factory signature checks, and empty capability checks as explicit rules.
- Do not hide contract failures behind generic `Exception` messages when the failure is expected.

### 5. Protect extension ergonomics

- Keep the manifest authoring path simple enough for new integrations.
- Avoid over-designing the contract when one new field or helper function would be enough.
- Prefer additive evolution of the manifest contract when backward compatibility matters.

## Strong Signals to Use This Skill

- A PR touches `fedot/extensions/contracts.py`, `registry.py`, `parameter_rules.py`, `operation_rules.py`, or `runtime_rules.py`.
- A change adds or modifies `FEDOT_EXTENSION_MANIFEST`.
- An external model should be available without editing several unrelated internal registries.
- Runtime integration depends on capabilities, hyperparameter schema, or manifest-driven operation resolution.

## Output Expectations

- Keep the extension contract explicit and discoverable.
- Preserve the canonical registration and smoke-test flow.
- Add or update mirrored tests under `tests/extensions`.
- Keep runtime behavior downstream of a validated manifest.

## References

- Read [contract-map.md](references/contract-map.md) for current entities and files.
- Read [implementation-flow.md](references/implementation-flow.md) for the canonical lifecycle.
- Read [test-and-review.md](references/test-and-review.md) for extension-specific coverage and review rules.