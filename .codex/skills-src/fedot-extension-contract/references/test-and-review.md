# Test and Review Guidance

## Core Test Targets

Prefer tests in `tests/extensions` for:

- valid manifest acceptance
- duplicate rejection
- lookup behavior after registration
- module-based manifest discovery
- smoke-test behavior for valid and broken factories
- parameter resolution from specs
- operation discovery from registered extensions

## Minimum Coverage for a Contract Change

When the contract or registry changes, keep coverage for:

- one happy path registration
- one invalid manifest path
- one duplicate or conflicting path
- one runtime-facing lookup or smoke-test path

## Review Questions

Ask:

1. Can a new extension still be integrated through one obvious path?
2. Did the change preserve validated data structures as the source of truth?
3. Are contract failures explicit and typed enough to test?
4. Did runtime behavior start depending on raw manifest blobs again?
5. Are extension tests mirrored and focused on behavior rather than internals?

## Typical Findings

Common review findings in this area:

- runtime code re-parses fields that should already be validated
- duplicate detection is missing or split across layers
- smoke tests are weakened or bypassed
- manifest evolution breaks old extension authors without need