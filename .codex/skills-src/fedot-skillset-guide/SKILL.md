---
name: fedot-skillset-guide
description: Choose, combine, and sequence the FEDOT team skills for repository work, including pure core extraction, invariant-oriented testing and review, safe configs, extension contracts, typed domain errors, and routing. Use when a task touches several architectural concerns, when onboarding teammates to the FEDOT skillset, or when deciding which skill should lead a refactor, review, or implementation request.
---

# FEDOT Skillset Guide

## Overview

Use this skill as the team-level guide for the FEDOT skillset. It explains which specialized skill should lead a task, which companion skills to add, and what order usually keeps refactors small, testable, and aligned with the repository's FP-informed direction.

## Quick Start

1. Read [skill-catalog.md](references/skill-catalog.md) to see the role of each FEDOT skill.
2. Read [composition-playbook.md](references/composition-playbook.md) to combine skills without overlap.
3. Read [team-workflow.md](references/team-workflow.md) for a repeatable task flow.
4. Read [smoke-suite.md](references/smoke-suite.md) when evolving the skillset and checking that routing still makes sense.

## Workflow

### 1. Pick the primary concern

Decide whether the task is mainly about:

- refactoring shell versus pure core boundaries
- modeling explicit states or failures
- making config parsing safe
- integrating external models
- testing and review quality
- choosing among several of the above

### 2. Choose one lead skill

Use one lead skill for the dominant concern:

- `$fedot-pure-core-shell`
- `$fedot-invariant-tests-review`
- `$fedot-safe-configs`
- `$fedot-extension-contract`
- `$fedot-typed-domain-errors`
- `$fedot-refactor-router` when the right lead is unclear

### 3. Add companion skills carefully

Add companion skills only when they shape the implementation, not just because they are related. Most tasks should need one lead skill and zero to two companions.

### 4. Sequence the work

Typical order:

1. route and define the domain model if needed
2. redesign parsing or pure logic boundaries
3. adjust shell or integration code
4. add or review tests

## Output Expectations

When using this skill, produce:

- the lead skill
- any companion skills
- one short reason for each choice
- a compact implementation order

## References

- Read [skill-catalog.md](references/skill-catalog.md) for the full catalog.
- Read [composition-playbook.md](references/composition-playbook.md) for common combinations.
- Read [team-workflow.md](references/team-workflow.md) for a repeatable process.
- Read [smoke-suite.md](references/smoke-suite.md) for maintenance checks.