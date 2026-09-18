# Tests Layout

The target test layout mirrors the production package layout under `fedot/`.

Examples:
- `fedot/api/main.py` -> `tests/api/test_main.py`
- `fedot/core/data/...` -> `tests/core/data/...`
- `fedot/core/pipelines/...` -> `tests/core/pipelines/...`
- `fedot/extensions/...` -> `tests/extensions/...`

## Rules

- Prefer `tests/` for all new and migrated tests.
- Keep `test/` as a temporary legacy location during the migration window only.
- Express test kind via pytest markers, not by directory name.
- Use `@pytest.mark.unit` for pure rules and narrow OOP-shell contracts.
- Use `@pytest.mark.integration` for subsystem and end-to-end behaviour.
- Use `@pytest.mark.property` for invariant and determinism checks.
- Use `@pytest.mark.slow` only when the scenario is materially expensive.

## Migration strategy

- Add new tests to `tests/` first.
- Mirror legacy coverage cluster-by-cluster instead of one large move.
- Remove legacy `test/` copies only after the mirrored path is stable in CI.
