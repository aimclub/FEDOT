
# Repo Boundaries

## Purpose

Use this file when deciding what to preserve as the effectful shell and what to extract into a pure helper.

## Boundary Objects to Preserve

### `fedot/api`

Preserve public coordinators and entry points such as:

- `Fedot`
- `FedotBuilder`
- `ApiDataProcessor`
- `ApiComposer`
- `PredefinedModel`
- `ApiParamsRepository`
- `ApiParams`
- `InputAnalyser`
- assumptions, preset, and filter builders or handlers

Treat these as shell objects that own lifecycle, compatibility, and orchestration.

### `fedot/core`

Preserve public graph and composition boundaries such as:

- `PipelineNode`
- `Pipeline`
- `PipelineBuilder`
- `PipelineTemplate`
- `PipelineAdapter`
- factory layers
- operation hierarchies
- `EvaluationStrategy`
- `Composer`
- `ComposerBuilder`
- objective or splitter abstractions

Move rule-heavy logic behind these boundaries into smaller collaborators where possible.

### `fedot/preprocessing`

Preserve data-entry and orchestration objects. Extract:

- source planning
- merge planning
- optional preprocessing decisions
- deterministic transform pipelines

### `fedot/extensions`

Preserve extension-facing registration and runtime boundaries. Extract:

- typed parameter resolution
- capability interpretation
- registry query logic
- operation discovery rules

### `fedot/remote`

Preserve integration boundaries and transport-facing APIs. Extract:

- parsing
- validation
- normalization
- safe defaulting

## What Belongs in the Pure Core

Prefer extraction when logic:

- derives a result from explicit inputs
- can be expressed as a deterministic function
- only needs data, not live dependencies
- is easy to validate independently
- represents business rules more than orchestration

Typical pure outputs:

- plans
- normalized parameter objects
- filtered candidate lists
- routing decisions
- capability sets
- typed validation results

## What Should Stay in the Shell

Keep these in the shell:

- file access
- network calls
- logging
- environment inspection
- runtime setup
- dependency construction
- compatibility glue
- command sequencing across effectful dependencies

## Python-Oriented Typing Guidance

Translate FP-informed patterns into Python carefully:

- Use `dataclass` for stable records.
- Use `Enum` for closed state sets.
- Use `Protocol` or narrow abstract base classes for dependency boundaries when helpful.
- Use `TypedDict` sparingly for raw external payloads.
- Avoid carrying `dict[str, Any]` across multiple layers when a shape is stable enough to name.