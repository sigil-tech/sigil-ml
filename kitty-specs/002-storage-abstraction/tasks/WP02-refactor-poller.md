---
work_package_id: WP02
title: Refactor Poller to Use DataStore
lane: planned
dependencies: [WP01]
subtasks:
- T008
- T009
- T010
- T011
- T012
- T013
phase: Phase 2 - Core Refactor
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
- FR-005
- FR-011
---

# Work Package Prompt: WP02 -- Refactor Poller to Use DataStore

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

This WP depends on WP01. Create the worktree with:

```bash
spec-kitty implement WP02 --base WP01
```

---

## Objectives & Success Criteria

- Refactor `EventPoller` to accept a `DataStore` instance instead of a `db_path`.
- Eliminate **all** direct `sqlite3` usage from `src/sigil_ml/poller.py`.
- The poller's behavior must remain functionally identical -- same poll logic, same prediction cadence, same data flow.
- **Success**: `import sqlite3` does not appear in `poller.py`. The `EventPoller` works with any `DataStore` implementation (including a mock).

## Context & Constraints

- **Spec**: `kitty-specs/002-storage-abstraction/spec.md` -- FR-005, FR-011
- **Prerequisite**: WP01 must be complete. `DataStore` protocol and `SqliteStore` exist in `src/sigil_ml/store.py` and `src/sigil_ml/store_sqlite.py`.
- **Current state of `poller.py`**: The `EventPoller` takes `db_path: Path` in its constructor. It creates a `sqlite3.Connection` per poll cycle via `_connect()`. The connection is passed through to `_predict_and_write`, `_session_info`, `_quality_features`, and `_write`. All SQL is inline.
- **Transaction pattern**: Currently, `_poll_once` opens one connection and commits once at the end (covering cursor update + predictions + audit). The DataStore methods each manage their own connections. This is acceptable because:
  - The cursor update and predictions are independent writes.
  - If the process crashes mid-cycle, the cursor won't advance, and the next poll will re-process the same events (idempotent design).
  - The slight behavior difference (multiple commits vs one) has no user-visible impact.

## Subtasks & Detailed Guidance

### Subtask T008 -- Change EventPoller constructor to accept DataStore

- **Purpose**: Replace the `db_path` parameter with a `DataStore` dependency, establishing the new interface.
- **Steps**:
  1. Open `src/sigil_ml/poller.py`.
  2. Add import: `from sigil_ml.store import DataStore`
  3. Change the constructor:

     **Before**:
     ```python
     def __init__(self, db_path: Path, models: dict) -> None:
         self.db_path = db_path
         ...
     ```

     **After**:
     ```python
     def __init__(self, store: DataStore, models: dict) -> None:
         self.store = store
         ...
     ```

  4. Remove the `db_path` attribute. The `self.db_path` reference in the `run()` log message should use a string representation of the store (or be updated to log "poller: started").
  5. **Important**: The poller currently passes `self.db_path` to `extract_stuck_features` and `extract_duration_features` in `_predict_and_write`. For now, these feature extraction functions still take `db_path`. This will be fixed in WP03. As a temporary bridge, the `SqliteStore` should expose a `db_path` property, OR the poller should get `db_path` from config. Prefer the latter to keep the store interface clean.

- **Files**: `src/sigil_ml/poller.py` (modify constructor and class-level attribute)
- **Parallel?**: No -- this is the first change; all other subtasks depend on it.
- **Notes**: Do not change the `models` parameter. Only the data access path changes.

### Subtask T009 -- Refactor `_poll_once` for DataStore

- **Purpose**: Replace the inline SQLite cursor read, event query, and cursor update with DataStore method calls.
- **Steps**:
  1. Replace the cursor read:

     **Before**:
     ```python
     cursor_id = conn.execute("SELECT last_event_id FROM ml_cursor WHERE id = 1").fetchone()
     since = cursor_id[0] if cursor_id else 0
     ```

     **After**:
     ```python
     since = self.store.get_cursor()
     ```

  2. Replace the event query:

     **Before**:
     ```python
     rows = conn.execute(
         "SELECT id, kind, source, payload, ts FROM events WHERE id > ? ORDER BY id ASC LIMIT 100",
         (since,),
     ).fetchall()
     ```

     **After**:
     ```python
     rows = self.store.get_events_since(since, limit=100)
     ```

     Note: The returned rows are already dicts with parsed payload (DataStore handles this).

  3. Update the event processing loop. Currently it converts tuples to dicts via `dict(zip(...))`. With DataStore, rows are already dicts, so simplify:

     ```python
     if not rows:
         return

     events = []
     for e in rows:
         # Payload already parsed by DataStore
         classification = self.activity.classify(e)
         e["_category"] = classification["category"]
         e["_category_confidence"] = classification["confidence"]
         events.append(e)
     ```

  4. Replace the cursor update:

     **Before**:
     ```python
     conn.execute(
         "UPDATE ml_cursor SET last_event_id = ?, updated_at = ? WHERE id = 1",
         (max_id, int(time.time() * 1000)),
     )
     ```

     **After**:
     ```python
     self.store.update_cursor(max_id)
     ```

  5. Remove the `conn` variable from `_poll_once` entirely. The method should no longer open or close any connection.

  6. Update `_predict_and_write` call -- it currently receives `conn`. Change to pass no connection:
     ```python
     self._predict_and_write()
     ```

  7. Remove the `conn.commit()` at the end of `_poll_once` (DataStore methods commit internally).

- **Files**: `src/sigil_ml/poller.py` (modify `_poll_once`)
- **Parallel?**: Yes -- can proceed once T008 is done.
- **Notes**: The `_poll_once` method currently catches `sqlite3.OperationalError` in the `run()` loop. After refactoring, the DataStore may raise different exceptions. Update the error handling in `run()` to catch `Exception` more broadly or the specific exceptions the DataStore raises.

### Subtask T010 -- Refactor `_predict_and_write` for DataStore

- **Purpose**: Replace inline SQL for getting active task, writing predictions, and writing audit events.
- **Steps**:
  1. Change the method signature -- remove the `conn` parameter:

     **Before**: `def _predict_and_write(self, conn: sqlite3.Connection) -> None:`
     **After**: `def _predict_and_write(self) -> None:`

  2. Replace active task lookup:

     **Before**:
     ```python
     task_id = self._active_task_id(conn)
     ```

     **After**:
     ```python
     task_id = self.store.get_active_task()
     ```

  3. Replace prediction writes (the `_write` helper):

     **Before**:
     ```python
     self._write(conn, "stuck", result, ...)
     ```

     **After**:
     ```python
     self.store.insert_prediction("stuck", result, confidence, ttl_sec)
     ```

     Replace all 5 calls to `self._write(conn, ...)` with `self.store.insert_prediction(...)`.

  4. Replace the audit event write at the end:

     **Before**:
     ```python
     conn.execute(
         "INSERT INTO ml_events (kind, endpoint, routing, latency_ms, ts) VALUES (...)",
         (latency_ms, int(time.time() * 1000)),
     )
     ```

     **After**:
     ```python
     self.store.insert_ml_event("prediction", "poller", "local", latency_ms)
     ```

  5. For stuck prediction, the poller calls `extract_stuck_features(self.db_path, task_id)`. Until WP03 changes the signature, use `config.db_path()` as a temporary bridge:
     ```python
     from sigil_ml import config
     # Temporary until WP03 updates feature extraction signatures
     feats = extract_stuck_features(config.db_path(), task_id)
     ```

  6. Same for `extract_duration_features(self.db_path, task_id)` -- use `config.db_path()` temporarily.

- **Files**: `src/sigil_ml/poller.py` (modify `_predict_and_write`)
- **Parallel?**: Yes -- can proceed once T008 is done.
- **Notes**: The 5 prediction writes are: stuck, activity, suggest (workflow), duration, quality.

### Subtask T011 -- Refactor `_session_info` and `_quality_features` for DataStore

- **Purpose**: Replace the remaining inline SQL in helper methods.
- **Steps**:
  1. Refactor `_session_info`:

     **Before**: Takes `conn: sqlite3.Connection` and `task_id`:
     ```python
     def _session_info(self, conn: sqlite3.Connection, task_id: str | None) -> dict:
         ...
         row = conn.execute("SELECT started_at, phase, test_fails FROM tasks WHERE id = ?", (task_id,)).fetchone()
     ```

     **After**: Remove `conn` parameter, use DataStore:
     ```python
     def _session_info(self, task_id: str | None) -> dict:
         session_elapsed_min = 0.0
         task_phase = None
         test_failures = 0

         if task_id:
             info = self.store.get_task_session_info(task_id)
             if info:
                 started_at = info.get("started_at", 0) or 0
                 session_elapsed_min = (time.time() * 1000 - started_at) / 60000.0
                 task_phase = info.get("phase")
                 test_failures = info.get("test_fails", 0) or 0

         return {
             "session_elapsed_min": max(session_elapsed_min, 0.0),
             "task_phase": task_phase,
             "test_failures": test_failures,
         }
     ```

  2. Refactor `_quality_features`:

     **Before**: Takes `conn: sqlite3.Connection`:
     ```python
     def _quality_features(self, conn: sqlite3.Connection) -> dict:
         ...
         row = conn.execute("SELECT test_runs, test_fails, commit_count FROM tasks ...").fetchone()
     ```

     **After**: Remove `conn` parameter, use DataStore:
     ```python
     def _quality_features(self) -> dict:
         ...
         stats = self.store.get_quality_task_stats()
         if stats and stats.get("test_runs"):
             test_pass_rate = 1.0 - (stats["test_fails"] / max(stats["test_runs"], 1))
             baseline_commits = max(stats["commit_count"], 1)
         else:
             test_pass_rate = 0.7
             baseline_commits = 1
         ...
     ```

  3. Update all callers in `_predict_and_write` to call these without `conn`.

- **Files**: `src/sigil_ml/poller.py` (modify `_session_info`, `_quality_features`)
- **Parallel?**: Yes -- can proceed once T008 is done.
- **Notes**: The `_quality_features` method also uses `self._buffer` for event window analysis. This part stays unchanged -- only the SQL query is replaced.

### Subtask T012 -- Remove `_connect`, `_write`, `_active_task_id`, and sqlite3 import

- **Purpose**: Clean up the poller module by removing all SQLite-specific code that has been replaced by DataStore calls.
- **Steps**:
  1. Delete the `_connect(self) -> sqlite3.Connection` method entirely.
  2. Delete the `_write(self, conn, model, result, confidence, ttl_sec)` helper method (replaced by `store.insert_prediction`).
  3. Delete the `_active_task_id(self, conn)` method (replaced by `store.get_active_task()`).
  4. Remove `import sqlite3` from the top of the file.
  5. Remove `from pathlib import Path` if no longer used.
  6. Verify the module still imports correctly: `python -c "from sigil_ml.poller import EventPoller"`

- **Files**: `src/sigil_ml/poller.py` (remove methods and imports)
- **Parallel?**: No -- do this after T009-T011 are complete to avoid breaking intermediate states.
- **Notes**: After this subtask, `poller.py` should have zero references to `sqlite3`.

### Subtask T013 -- Update app.py startup to pass DataStore to EventPoller

- **Purpose**: Wire the new `DataStore` into the application startup sequence so the poller receives a store instance.
- **Steps**:
  1. Open `src/sigil_ml/app.py`.
  2. In `startup_event()`, create a store before creating the poller:

     **Before**:
     ```python
     db = config.db_path()
     ensure_ml_tables(db)
     ...
     state.poller = EventPoller(db_path=db, models={...})
     ```

     **After**:
     ```python
     from sigil_ml.store import create_store
     store = create_store()
     store.ensure_tables()
     ...
     state.poller = EventPoller(store=store, models={...})
     ```

  3. The `TrainingScheduler` still takes `db_path` in this WP (it will be refactored in WP03). Keep: `scheduler = TrainingScheduler(db, reload_callback=...)`.
  4. Store the `store` instance on `state` for later use by routes (WP03): `state.store = store`.
  5. Update `AppState` to include `store: DataStore | None = None`.

- **Files**: `src/sigil_ml/app.py` (modify `startup_event`, `AppState`)
- **Parallel?**: No -- depends on T008-T012 being complete.
- **Notes**: The `ensure_ml_tables` import from `schema` can remain for now (removed in WP05). The key change is that the poller gets a `DataStore` instead of `db_path`.

## Risks & Mitigations

- **Risk**: The current poller uses a single connection per poll cycle for atomicity. With DataStore, each method opens its own connection. If the process crashes between cursor update and prediction write, the cursor may advance without predictions being written. **Mitigation**: This is acceptable because the poller is designed for eventual consistency -- predictions are ephemeral with TTLs, and the next cycle will generate new ones.
- **Risk**: The `run()` loop catches `sqlite3.OperationalError`. After refactoring, this exception type won't be raised directly. **Mitigation**: Update to catch `Exception` broadly or add a base `StoreError` exception in a follow-up.

## Review Guidance

- Verify `poller.py` has zero `sqlite3` imports or usage.
- Verify `EventPoller.__init__` takes `store: DataStore` (not `db_path`).
- Verify all 5 prediction writes go through `store.insert_prediction`.
- Verify `app.py` creates a store and passes it to the poller.
- Run existing tests and verify no regressions.

## Activity Log

- 2026-03-29T16:29:57Z -- system -- lane=planned -- Prompt created.
