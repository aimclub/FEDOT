# FEDOT Skillset Smoke Suite

This file is the repository-level maintenance checklist for the installed FEDOT skills.

## How to Use

1. Pick 3-5 prompts from `.codex/skills-src/smoke-prompts.md`.
2. For routing checks, compare the result against `fedot-skillset-guide/references/smoke-suite.md`.
3. If a prompt now routes to a different lead skill, decide whether the skill descriptions improved or drifted.
4. Update the relevant `SKILL.md` and `agents/openai.yaml` before syncing skills again.

## Minimum Regression Pass

Run at least these checks after meaningful edits:

- one pure-core extraction prompt
- one safe-config prompt
- one extension-contract prompt
- one typed-domain-errors prompt
- one review or invariant-test prompt
- one ambiguous router prompt

## Sync Reminder

After editing skills in `.codex/skills-src`, sync them with:

```powershell
powershell -ExecutionPolicy Bypass -File .\.codex\scripts\sync-skills.ps1 -SourceRoot .\.codex\skills-src -TargetRoot C:\Users\user\.codex\skills
```