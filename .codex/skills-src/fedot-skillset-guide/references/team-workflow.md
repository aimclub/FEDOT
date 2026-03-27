# Team Workflow

## Suggested Team Routine

### 1. Start with one sentence of task classification

Before coding, state the dominant concern in one sentence. This reduces drift and helps pick the lead skill.

### 2. Use one lead skill and minimal companions

Do not turn every task into a multi-skill exercise. Prefer one lead skill and only add companions that materially change the solution.

### 3. Keep refactors small and vertical

For FEDOT, prefer vertical slices that leave public boundaries working while improving one concern deeply:

- one planner
- one parser
- one registry path
- one preprocessing flow

### 4. Mirror tests as you go

Whenever a refactor lands, update the mirrored `tests/` area in the same pass. Do not postpone test realignment until later.

### 5. Re-run smoke prompts after editing the skillset

When the skillset changes, use the smoke suite to confirm that routing and scope still feel right.

## Good Outcomes

A good use of the skillset should leave:

- a clear lead skill choice
- less hidden branching
- more explicit domain meaning
- safer parsing or extension behavior where relevant
- tests that match the changed architecture