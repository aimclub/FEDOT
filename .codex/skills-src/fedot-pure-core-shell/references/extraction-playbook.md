# Extraction Playbook

## Workflow

### 1. Mark the shell

Locate the class or function that owns:

- lifecycle
- dependency wiring
- side effects
- entry-point compatibility

Do not dissolve that boundary unless the user asks for an API redesign.

### 2. Mark the rule set

Identify logic that decides or transforms:

- which mode to run
- which defaults to apply
- which backend or operation to choose
- how to normalize parameters
- how to filter candidate operations
- how to build a preprocessing or execution plan

These are the first extraction candidates.

### 3. Freeze explicit inputs

Before moving code, name the real inputs:

- raw params
- existing state
- discovered capabilities
- repository contents
- user flags
- defaults

If the helper would need to reach back into object state, first pass that state explicitly.

### 4. Define typed outputs

Return one of:

- a typed plan
- a normalized parameter object
- a filtered result set
- a decision enum
- a structured validation result

Avoid returning half-normalized dictionaries when the shape is stable.

### 5. Move the logic

Extract the rule set into a small helper or module. Make the shell responsible only for:

- collecting raw inputs
- calling the helper
- applying effects using the helper result

### 6. Collapse scattered normalization

If the same defaults or conversions were spread across several methods, centralize them in the pure helper. Keep one canonical path.

### 7. Add focused tests

Write:

- unit tests for the new helper
- shell smoke tests for the preserved boundary
- invariants for deterministic rules when appropriate

## Recommended Shapes

### Pure function

Use when the logic is one deterministic transformation.

### Small module with a few helpers

Use when the logic has several stages such as parse, validate, normalize, and choose.

### Typed result object

Use when the shell needs several outputs, for example:

- normalized params
- warnings
- selected strategy

## Anti-Patterns

Avoid these moves:

- extracting only names while leaving branching scattered
- hiding effectful calls inside a helper now labeled "pure"
- keeping raw mutable shared state as an implicit dependency
- introducing a deep abstraction stack for one short rule
- converting clear procedural code into awkward combinator style for appearance only

## Local Mutation Rule

Allow local mutation inside a helper only when:

- it is fully encapsulated
- it does not leak through the external API
- it improves performance or clarity
- the caller still sees deterministic behavior