---
work_package_id: WP05
title: App Wiring, Config & Cleanup
lane: planned
dependencies: [WP01, WP02, WP03, WP04]
subtasks:
- T029
- T030
- T031
- T032
- T033
phase: Phase 4 - Integration & Polish
assignee: ''
agent: ''
shell_pid: ''
review_status: ''
reviewed_by: ''
history:
- timestamp: '2026-03-29T16:29:57Z'
  lane: planned
  agent: system
  shell_pid: ''
  action: Prompt generated via /spec-kitty.tasks
requirement_refs:
- FR-008
- FR-009
- FR-010
---

# Work Package Prompt: WP05 -- App Wiring, Config & Cleanup

## Important: Review Feedback Status

**Read this first if you are implementing this task!**

- **Has review feedback?**: Check the `review_status` field above. If it says `has_feedback`, scroll to the **Review Feedback** section immediately.
- **You must address all feedback** before your work is complete.
- **Mark as acknowledged**: When you understand the feedback and begin addressing it, update `review_status: acknowledged` in the frontmatter.

---

## Review Feedback

> **Populated by `/spec-kitty.review`** -- Reviewers add detailed feedback here when work needs changes.

*[This section is empty initially.]*

---

## Markdown Formatting
Wrap HTML/XML tags in backticks: `` `<div>` ``, `` `<script>` ``
Use language identifiers in code blocks: ````python`, ````bash`

---

## Implementation Command

This WP depends on all prior WPs. Create the worktree with:

```bash
spec-kitty implement WP05 --base WP04
```

Or if WP03 was the last merged:

```bash
spec-kitty implement WP05 --base WP03
```

---

## Objectives & Success Criteria

- Complete end-to-end integration: `create_app()` selects the correct backend based on operating mode.
- Configuration supports both SQLite (local) and Postgres (cloud) backends via environment variables.
- Retire `schema.py` -- its functionality is now in `DataStore.ensure_tables()`.
- **Success criteria SC-002**: `grep -r "import sqlite3" src/sigil_ml/` returns ONLY `store_sqlite.py`.
- **Success criteria SC-004**: Switching from SQLite to Postgres requires only setting `SIGIL_MODE=cloud` and `SIGIL_POSTGRES_URL=...` -- no code changes.
- Start sigil-ml in local mode -- all behavior identical to pre-refactor.

## Context & Constraints

- **Spec**: `kitty-specs/002-storage-abstraction/spec.md` -- FR-008, FR-009, FR-010
- **Prerequisites**: WP01-WP04 are all complete. The DataStore protocol, both implementations, and all module refactors are done.
- **Current state after WP01-WP04**:
  - `store.py` has `DataStore` protocol and `create_store()` factory.
  - `store_sqlite.py` has `SqliteStore`.
  - `store_postgres.py` has `PostgresStore`.
  - `poller.py`, `features.py`, `routes.py`, `trainer.py`, `scheduler.py` all use DataStore.
  - `schema.py` still exists but is no longer called (replaced by `store.ensure_tables()`).
  - `config.py` may have been partially updated in WP04 T028.
  - `app.py` creates a store and passes it to all consumers.
- **Feature 001 dependency**: Cloud serving mode (feature 001) introduces `--mode cloud` CLI flag. If not yet merged, `SIGIL_MODE` environment variable is the primary mechanism.

## Subtasks & Detailed Guidance

### Subtask T029 -- Finalize config.py with operating_mode() and postgres_url()

- **Purpose**: Ensure the config module has clean, well-documented functions for all DataStore-related configuration.
- **Steps**:
  1. Open `src/sigil_ml/config.py`.
  2. If T028 (WP04) already added these functions, verify they are correct. If not, add them:

     ```python
     def operating_mode() -> str:
         """Return the operating mode: 'local' or 'cloud'.

         Set via SIGIL_MODE environment variable. Defaults to 'local'.
         In 'cloud' mode, a Postgres backend is used instead of SQLite.
         """
         return os.environ.get("SIGIL_MODE", "local")

     def postgres_url() -> str | None:
         """Return the Postgres connection URL for cloud mode.

         Set via SIGIL_POSTGRES_URL environment variable.
         Required when operating_mode() == 'cloud'.
         Format: postgresql://user:password@host:port/database
         """
         return os.environ.get("SIGIL_POSTGRES_URL")

     def tenant_id() -> str:
         """Return the tenant identifier for multi-tenant cloud deployment.

         Set via SIGIL_TENANT environment variable. Defaults to 'default'.
         Used to create per-tenant Postgres schemas.
         """
         return os.environ.get("SIGIL_TENANT", "default")
     ```

  3. Verify `db_path()` still works for local mode -- it should remain unchanged.
  4. Verify `create_store()` in `store.py` uses these config functions (not raw `os.environ`).

- **Files**: `src/sigil_ml/config.py` (verify/add ~25 lines)
- **Parallel?**: Yes -- independent of other subtasks.
- **Notes**: Keep the functions simple. No validation here -- validation happens in `create_store()`.

### Subtask T030 -- Finalize app.py create_app() wiring

- **Purpose**: Ensure the application startup sequence is clean, creates the store once, and passes it to all consumers.
- **Steps**:
  1. Open `src/sigil_ml/app.py`.
  2. Verify the complete startup flow in `startup_event()`:

     ```python
     @application.on_event("startup")
     async def startup_event() -> None:
         from sigil_ml.store import create_store

         store = create_store()

         try:
             store.ensure_tables()
         except Exception:
             logger.warning("schema bootstrap failed (sigild may not have started yet)", exc_info=True)

         state.store = store
         state.load_models()

         state.poller = EventPoller(
             store=store,
             models={
                 "stuck": state.stuck,
                 "activity": state.activity,
                 "workflow": state.workflow,
                 "duration": state.duration,
                 "quality": state.quality,
             },
         )
         asyncio.create_task(state.poller.run())

         scheduler = TrainingScheduler(store, reload_callback=state.reload_models_into_poller)

         async def _schedule_loop():
             while True:
                 await asyncio.get_event_loop().run_in_executor(None, scheduler.check_and_retrain)
                 await asyncio.sleep(600)

         asyncio.create_task(_schedule_loop())

         logger.info("sigil-ml: models loaded, poller started, scheduler active")
     ```

  3. Remove any remaining references to `config.db_path()` in the startup sequence (it should be encapsulated inside `create_store()`).
  4. Remove the `from sigil_ml.schema import ensure_ml_tables` import if it still exists.
  5. Verify shutdown still works: `state.poller.stop()` is unchanged.

- **Files**: `src/sigil_ml/app.py` (verify/finalize ~10 lines of changes)
- **Parallel?**: No -- do after verifying all WP01-WP04 changes are in place.
- **Notes**: The key change is that `create_store()` is the single point of backend selection. No other code needs to know whether SQLite or Postgres is being used.

### Subtask T031 -- Retire schema.py

- **Purpose**: `schema.py` defined `ensure_ml_tables()` which is now replaced by `DataStore.ensure_tables()`. The module should be removed.
- **Steps**:
  1. First, verify no code imports from `schema.py`:
     ```bash
     grep -r "from sigil_ml.schema" src/sigil_ml/
     grep -r "from sigil_ml import schema" src/sigil_ml/
     grep -r "import schema" src/sigil_ml/
     ```
  2. Also check tests:
     ```bash
     grep -r "schema" tests/
     ```
  3. If no imports remain, delete `src/sigil_ml/schema.py`.
  4. If tests import it, update them to use `SqliteStore.ensure_tables()` instead.
  5. Verify the project still imports cleanly:
     ```bash
     python -c "import sigil_ml; from sigil_ml.app import create_app"
     ```

- **Files**: `src/sigil_ml/schema.py` (delete)
- **Parallel?**: Yes -- independent of other subtasks, but verify imports first.
- **Notes**: If other features or tests still reference `schema.py`, keep it with a deprecation comment rather than breaking things. The key requirement is that `app.py` no longer calls it.

### Subtask T032 -- Final sqlite3 import audit

- **Purpose**: Verify success criteria SC-002: no module outside DataStore implementations imports sqlite3.
- **Steps**:
  1. Run the audit:
     ```bash
     grep -rn "import sqlite3" src/sigil_ml/
     ```
  2. **Expected result**: Only `src/sigil_ml/store_sqlite.py` should appear.
  3. If any other file appears:
     - If it's `schema.py` and T031 hasn't deleted it yet, that's expected -- it will be removed.
     - If it's any other module, there's a missed refactoring. Fix it.
  4. Also check for indirect sqlite3 usage:
     ```bash
     grep -rn "sqlite3\." src/sigil_ml/
     ```
  5. Document the results.

- **Files**: None (audit only)
- **Parallel?**: No -- do this after all other subtasks in this WP are complete.
- **Notes**: This is a validation step, not a code change (unless issues are found).

### Subtask T033 -- Finalize AppState to hold DataStore reference

- **Purpose**: Ensure `AppState` has a clean `store` attribute typed as `DataStore`.
- **Steps**:
  1. Open `src/sigil_ml/app.py`.
  2. Verify `AppState` has the `store` attribute:

     ```python
     from sigil_ml.store import DataStore

     class AppState:
         """Holds model instances and runtime state, passed to routes."""

         def __init__(self) -> None:
             self.store: DataStore | None = None
             self.stuck: StuckPredictor | None = None
             self.activity: ActivityClassifier | None = None
             self.workflow: WorkflowStatePredictor | None = None
             self.duration: DurationEstimator | None = None
             self.quality: QualityEstimator | None = None
             self.poller: EventPoller | None = None
             self.training_in_progress: bool = False
     ```

  3. Verify `reload_models_into_poller` still works correctly -- it accesses `self.poller` which still exists.
  4. Consider whether `reload_models_into_poller` should also update the store reference on the poller (it shouldn't need to -- the store doesn't change, only models do).

- **Files**: `src/sigil_ml/app.py` (verify/modify AppState)
- **Parallel?**: Yes -- can be done alongside T029-T031.
- **Notes**: This may already be in place from WP02 T013. Verify and clean up.

## Risks & Mitigations

- **Risk**: `schema.py` deletion breaks an import in tests or another module that wasn't caught. **Mitigation**: Comprehensive grep before deletion (T031 steps 1-2).
- **Risk**: Feature 001 (cloud serving mode) introduces conflicting `--mode` CLI logic. **Mitigation**: The config functions use environment variables as the primary mechanism, which is compatible with any CLI wrapper.
- **Risk**: Post-refactor behavior differs subtly from pre-refactor (e.g., connection timing, error handling). **Mitigation**: Run the full test suite and manual smoke test.

## Review Guidance

- **Critical check**: `grep -r "import sqlite3" src/sigil_ml/` returns ONLY `store_sqlite.py`.
- Verify `schema.py` is deleted (or deprecated with a clear TODO).
- Verify `create_app()` startup uses `create_store()` and passes the store to all consumers.
- Verify `config.py` has `operating_mode()`, `postgres_url()`, and `tenant_id()`.
- Manual smoke test: `python -c "from sigil_ml.app import create_app; app = create_app()"` should succeed in local mode.
- Verify environment variable switching works: `SIGIL_MODE=cloud` should attempt Postgres (will fail without a URL, but should fail with a clear `ValueError`, not an import error).

## Activity Log

- 2026-03-29T16:29:57Z -- system -- lane=planned -- Prompt created.
