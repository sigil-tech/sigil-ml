---
work_package_id: "WP03"
title: "Refactor Features, Routes & Training to Use DataStore"
lane: "planned"
dependencies: ["WP01", "WP02"]
subtasks:
  - "T011"
  - "T012"
  - "T013"
  - "T014"
  - "T015"
  - "T016"
phase: "Phase 2 - Core Refactor"
assignee: ""
agent: ""
shell_pid: ""
review_status: ""
reviewed_by: ""
history:
  - timestamp: "2026-03-30T01:45:06Z"
    lane: "planned"
    agent: "system"
    shell_pid: ""
    action: "Prompt generated via /spec-kitty.tasks"
requirement_refs:
  - "FR-005"
  - "FR-006"
  - "FR-007"
  - "FR-011"
---

# Work Package Prompt: WP03 -- Refactor Features, Routes & Training to Use DataStore

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

## Objectives & Success Criteria

- Eliminate all direct `sqlite3` usage from `features.py`, `routes.py`, `training/trainer.py`, and `training/scheduler.py`.
- After this WP, `grep -r "import sqlite3" src/sigil_ml/` should return ONLY `store_sqlite.py`.
- All feature extraction, route handling, and training code must use the `DataStore` interface exclusively.
- All existing functionality must remain identical -- same predictions, same responses, same training behavior.

**Success criteria**:
- FR-006: API route handlers use the DataStore protocol for all data access.
- FR-007: Training modules use the DataStore protocol for reading training data and writing audit events.
- SC-002: No module outside of the DataStore implementations imports `sqlite3` directly.
- SC-003: Poller, routes, and training modules can operate against a mock DataStore in tests.

**Implementation command**: `spec-kitty implement WP03 --base WP02`

## Context & Constraints

- **Prerequisites**: WP01 (DataStore + SqliteStore) and WP02 (poller refactored) must be complete.
- **Spec**: `kitty-specs/002-storage-abstraction/spec.md` -- User Story 3 (all components use DataStore).
- **Current state**: 6 files still import `sqlite3` directly. After WP02, `poller.py` is clean. This WP addresses the remaining 4 files: `features.py`, `routes.py`, `training/trainer.py`, `training/scheduler.py`.

### Complete Call Site Inventory

**`features.py`** -- 2 sqlite3 connection opens:
1. `_query_task()` (line 72): `sqlite3.connect(str(db_path))` -- reads a task by ID
2. `_query_events_for_task()` (line 99): `sqlite3.connect(str(db_path))` -- reads events in a time window
3. `extract_stuck_features(db_path, task_id)` (line 119): calls `_query_task` and `_query_events_for_task`
4. `extract_duration_features(db_path, task_id)` (line 185): calls `_query_task` and `_query_events_for_task`

**`routes.py`** -- 3 direct sqlite3 usages:
5. `/status` endpoint (line 124): `sqlite3.connect(str(db))` -- reads cursor + predictions
6. `/predict/stuck` (line 153): calls `extract_stuck_features(config.db_path(), req.task_id)`
7. `/predict/duration` (line 198): calls `extract_duration_features(config.db_path(), req.task_id)`
8. `/train` (line 227): `_run_training(state, db)` passes db_path string
9. `_run_training` (line 235): `Trainer(db_path)` creates trainer with db_path

**`training/trainer.py`** -- 2 sqlite3 connection opens:
10. `_train_stuck()` (line 63): `sqlite3.connect(str(self.db_path))` -- reads completed task IDs
11. `_train_stuck()` (line 81): calls `extract_stuck_features(self.db_path, task_id)` per task
12. `_train_duration()` (line 104): `sqlite3.connect(str(self.db_path))` -- reads completed tasks with timestamps
13. `_train_duration()` (line 125): calls `extract_duration_features(self.db_path, task_id)` per task

**`training/scheduler.py`** -- 3 sqlite3 connection opens:
14. `__init__` (line 29): calls `self._count_completed()` which opens sqlite3
15. `_count_completed()` (line 57-58): `sqlite3.connect(str(self.db_path))` -- counts completed tasks
16. `_log_retrain()` (line 69-70): `sqlite3.connect(str(self.db_path))` -- inserts ml_events row

---

## Subtasks & Detailed Guidance

### Subtask T011 -- Refactor `features.py` for DataStore

- **Purpose**: Replace `_query_task` and `_query_events_for_task` with DataStore calls, and change the public API signatures of `extract_stuck_features` and `extract_duration_features` from `(db_path, task_id)` to `(store, task_id)`.
- **Files**: `src/sigil_ml/features.py`
- **Parallel?**: Yes -- can proceed in parallel with T012-T013 (routes) and T014-T015 (training).

**Steps**:
1. Remove `import sqlite3` (line 7) and `from pathlib import Path` (line 9, if only used for db_path).

2. Add import:
   ```python
   from sigil_ml.store import DataStore
   ```

3. Replace `_query_task(db_path, task_id)` (lines 70-81):

   **Before**:
   ```python
   def _query_task(db_path: str | Path, task_id: str) -> dict[str, Any] | None:
       conn = sqlite3.connect(str(db_path))
       conn.row_factory = sqlite3.Row
       try:
           cur = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
           row = cur.fetchone()
           if row is None:
               return None
           return dict(row)
       finally:
           conn.close()
   ```

   **After**:
   ```python
   def _query_task(store: DataStore, task_id: str) -> dict[str, Any] | None:
       return store.get_task_by_id(task_id)
   ```

   Or inline the `store.get_task_by_id()` call directly at call sites and remove `_query_task` entirely.

4. Replace `_query_events_for_task(db_path, task_id, since)` (lines 84-116):

   **Before**: Opens its own `sqlite3.connect`, queries task for time bounds, then queries events.

   **After**:
   ```python
   def _query_events_for_task(store: DataStore, task_id: str, since: int | None = None) -> list[dict[str, Any]]:
       return store.get_events_for_task(task_id, since=since)
   ```

   The DataStore's `get_events_for_task` already handles the time window logic and JSON payload parsing (implemented in WP01).

5. Update `extract_stuck_features` (line 119):

   **Before**:
   ```python
   def extract_stuck_features(db_path: str | Path, task_id: str) -> dict[str, float]:
       task = _query_task(db_path, task_id)
       ...
       events = _query_events_for_task(db_path, task_id)
   ```

   **After**:
   ```python
   def extract_stuck_features(store: DataStore, task_id: str) -> dict[str, float]:
       task = store.get_task_by_id(task_id)
       if task is None:
           return {
               "test_failure_count": 0.0,
               "time_in_phase_sec": 0.0,
               "edit_velocity": 0.0,
               "file_switch_rate": 0.0,
               "session_length_sec": 0.0,
               "time_since_last_commit_sec": 0.0,
           }
       events = store.get_events_for_task(task_id)
       # ... rest of feature extraction logic unchanged ...
   ```

6. Update `extract_duration_features` (line 185):

   **Before**:
   ```python
   def extract_duration_features(db_path: str | Path, task_id: str) -> dict[str, float]:
       task = _query_task(db_path, task_id)
       ...
       events = _query_events_for_task(db_path, task_id)
   ```

   **After**:
   ```python
   def extract_duration_features(store: DataStore, task_id: str) -> dict[str, float]:
       task = store.get_task_by_id(task_id)
       if task is None:
           return { ... }  # same defaults as current
       # ...
       events = store.get_events_for_task(task_id)
       # ... rest unchanged ...
   ```

7. Verify: `grep -n "sqlite3" src/sigil_ml/features.py` returns no results.

**Impact on callers** (to be updated in this WP):
- `poller.py` lines 140, 160: `extract_stuck_features(self.db_path, task_id)` -> `extract_stuck_features(self.store, task_id)` (may already be done in WP02 or needs update here)
- `routes.py` lines 153, 198: `extract_stuck_features(config.db_path(), req.task_id)` -> updated in T013
- `trainer.py` lines 81, 125: `extract_stuck_features(self.db_path, task_id)` -> updated in T014

---

### Subtask T012 -- Refactor `/status` endpoint to use DataStore

- **Purpose**: Remove direct `sqlite3` usage from the `/status` endpoint in `routes.py`.
- **Files**: `src/sigil_ml/routes.py`
- **Parallel?**: Yes -- can proceed in parallel with T011, T014-T015.

**Steps**:
1. The `/status` endpoint (lines 120-143) currently opens its own SQLite connection:

   **Before**:
   ```python
   @fastapi_app.get("/status")
   async def status() -> dict:
       db = config.db_path()
       try:
           conn = sqlite3.connect(str(db), timeout=5.0)
           conn.execute("PRAGMA busy_timeout=5000")
           conn.row_factory = sqlite3.Row
           try:
               cursor_row = conn.execute("SELECT last_event_id, updated_at FROM ml_cursor WHERE id = 1").fetchone()
               preds = conn.execute(
                   "SELECT model, confidence, created_at FROM ml_predictions "
                   "WHERE expires_at IS NULL OR expires_at > ? "
                   "ORDER BY created_at DESC",
                   (int(time.time() * 1000),),
               ).fetchall()
               return {
                   "cursor": dict(cursor_row) if cursor_row else None,
                   "latest_predictions": [dict(r) for r in preds],
                   "poller_running": state.poller is not None and state.poller._running,
               }
           finally:
               conn.close()
       except sqlite3.OperationalError:
           return {"cursor": None, "latest_predictions": [], "poller_running": False}
   ```

   **After**:
   ```python
   @fastapi_app.get("/status")
   async def status() -> dict:
       try:
           status_data = state.store.get_status_data()
           return {
               "cursor": status_data["cursor"],
               "latest_predictions": status_data["latest_predictions"],
               "poller_running": state.poller is not None and state.poller._running,
           }
       except Exception:
           return {"cursor": None, "latest_predictions": [], "poller_running": False}
   ```

2. This requires `state` (the `AppState` instance) to have a `store` attribute. This is set up in T016 (interim app wiring).

**Notes**:
- The `config.db_path()` import in routes.py can be removed after all endpoints are refactored.
- The `import sqlite3` can be removed after T012 + T013 are both done.

---

### Subtask T013 -- Refactor `/predict/*` and `/train` endpoints for DataStore

- **Purpose**: Update all prediction endpoints and the training endpoint to use DataStore instead of `config.db_path()`.
- **Files**: `src/sigil_ml/routes.py`
- **Parallel?**: Yes -- can proceed in parallel with T011, T014-T015.

**Steps**:
1. Update `/predict/stuck` (line 153):

   **Before**:
   ```python
   features = extract_stuck_features(config.db_path(), req.task_id)
   ```

   **After**:
   ```python
   features = extract_stuck_features(state.store, req.task_id)
   ```

2. Update `/predict/duration` (line 198):

   **Before**:
   ```python
   features = extract_duration_features(config.db_path(), req.task_id)
   ```

   **After**:
   ```python
   features = extract_duration_features(state.store, req.task_id)
   ```

3. Update `/train` endpoint (lines 222-228):

   **Before**:
   ```python
   db = req.db or str(config.db_path())
   background_tasks.add_task(_run_training, state, db)
   return TrainResponse(status="started", message=f"Training started with db={db}")
   ```

   **After**:
   ```python
   background_tasks.add_task(_run_training, state)
   return TrainResponse(status="started", message="Training started")
   ```

   Note: The `TrainRequest.db` field can be deprecated or removed. Training now uses the store already configured in state.

4. Update `_run_training` (lines 231-243):

   **Before**:
   ```python
   def _run_training(state: AppState, db_path: str) -> None:
       try:
           state.training_in_progress = True
           trainer = Trainer(db_path)
           ...
   ```

   **After**:
   ```python
   def _run_training(state: AppState) -> None:
       try:
           state.training_in_progress = True
           trainer = Trainer(state.store)
           ...
   ```

5. Remove `import sqlite3` (line 6) and `from sigil_ml import config` (line 12, if no longer used elsewhere in the file).

6. Remove the `import time` if only used for `int(time.time() * 1000)` in the old `/status` endpoint (check other usages first -- `_start_time` uses `time.time()`).

7. Verify: `grep -n "sqlite3\|config\.db_path" src/sigil_ml/routes.py` returns no results.

---

### Subtask T014 -- Refactor `training/trainer.py` to accept DataStore

- **Purpose**: Replace all `sqlite3` usage in the Trainer class with DataStore calls.
- **Files**: `src/sigil_ml/training/trainer.py`
- **Parallel?**: Yes -- can proceed in parallel with T011-T013, T015.

**Steps**:
1. Change the constructor:

   **Before** (line 23):
   ```python
   def __init__(self, db_path: str | Path) -> None:
       self.db_path = Path(db_path)
   ```

   **After**:
   ```python
   def __init__(self, store: DataStore) -> None:
       self.store = store
   ```

2. Add import:
   ```python
   from sigil_ml.store import DataStore
   ```

3. Remove `import sqlite3` (line 4) and `from pathlib import Path` (line 6).

4. Refactor `_train_stuck()` (lines 53-93):

   **Before** (lines 59-68):
   ```python
   if not self.db_path.exists():
       logger.warning("Database not found: %s", self.db_path)
       return 0

   conn = sqlite3.connect(str(self.db_path))
   conn.row_factory = sqlite3.Row
   try:
       rows = conn.execute("SELECT id FROM tasks WHERE completed_at IS NOT NULL").fetchall()
   finally:
       conn.close()
   ```

   **After**:
   ```python
   task_ids = self.store.get_completed_task_ids()
   ```

   Then at line 70:
   ```python
   if len(task_ids) < 10:
       ...
   ```

   And lines 78-82:
   ```python
   for task_id in task_ids:
       features = extract_stuck_features(self.store, task_id)
       ...
   ```

   Note: The `self.db_path.exists()` check is no longer needed -- the DataStore handles connection errors internally.

5. Refactor `_train_duration()` (lines 95-137):

   **Before** (lines 100-112):
   ```python
   if not self.db_path.exists():
       return 0

   conn = sqlite3.connect(str(self.db_path))
   conn.row_factory = sqlite3.Row
   try:
       rows = conn.execute(
           "SELECT id, started_at, completed_at FROM tasks "
           "WHERE completed_at IS NOT NULL AND started_at IS NOT NULL"
       ).fetchall()
   finally:
       conn.close()
   ```

   **After**:
   ```python
   rows = self.store.get_completed_tasks_with_timestamps()
   ```

   Then lines 114-129:
   ```python
   if len(rows) < 10:
       ...

   for row in rows:
       task_id = row["id"]
       features = extract_duration_features(self.store, task_id)
       ...
       duration_min = (row["completed_at"] - row["started_at"]) / 60000.0
       ...
   ```

6. Verify: `grep -n "sqlite3" src/sigil_ml/training/trainer.py` returns no results.

---

### Subtask T015 -- Refactor `training/scheduler.py` to accept DataStore

- **Purpose**: Replace all `sqlite3` usage in the TrainingScheduler class with DataStore calls.
- **Files**: `src/sigil_ml/training/scheduler.py`
- **Parallel?**: Yes -- can proceed in parallel with T011-T014.

**Steps**:
1. Change the constructor:

   **Before** (line 25):
   ```python
   def __init__(self, db_path: Path, reload_callback) -> None:
       self.db_path = db_path
       self._reload = reload_callback
       self._last_retrain: float = 0.0
       self._baseline_tasks: int = self._count_completed()
   ```

   **After**:
   ```python
   def __init__(self, store: DataStore, reload_callback) -> None:
       self.store = store
       self._reload = reload_callback
       self._last_retrain: float = 0.0
       self._baseline_tasks: int = self._count_completed()
   ```

2. Add import:
   ```python
   from sigil_ml.store import DataStore
   ```

3. Remove `import sqlite3` (line 4) and `from pathlib import Path` (line 6).

4. Update `check_and_retrain()` line 46:

   **Before**:
   ```python
   result = Trainer(self.db_path).train_all()
   ```

   **After**:
   ```python
   result = Trainer(self.store).train_all()
   ```

5. Refactor `_count_completed()` (lines 55-65):

   **Before**:
   ```python
   def _count_completed(self) -> int:
       try:
           conn = sqlite3.connect(str(self.db_path), timeout=5.0)
           conn.execute("PRAGMA busy_timeout=5000")
           try:
               row = conn.execute("SELECT COUNT(*) FROM tasks WHERE completed_at IS NOT NULL").fetchone()
               return row[0] if row else 0
           finally:
               conn.close()
       except sqlite3.OperationalError:
           return 0
   ```

   **After**:
   ```python
   def _count_completed(self) -> int:
       try:
           return self.store.count_completed_tasks()
       except Exception:
           return 0
   ```

6. Refactor `_log_retrain()` (lines 67-84):

   **Before**:
   ```python
   def _log_retrain(self, result: dict) -> None:
       try:
           conn = sqlite3.connect(str(self.db_path), timeout=5.0)
           conn.execute("PRAGMA busy_timeout=5000")
           try:
               conn.execute(
                   "INSERT INTO ml_events (kind, endpoint, routing, latency_ms, ts) "
                   "VALUES ('retrain', 'scheduler', 'local', ?, ?)",
                   (int(result.get("duration_sec", 0) * 1000), int(time.time() * 1000)),
               )
               conn.commit()
           finally:
               conn.close()
       except sqlite3.OperationalError:
           logger.warning("scheduler: failed to log retrain event")
   ```

   **After**:
   ```python
   def _log_retrain(self, result: dict) -> None:
       try:
           latency_ms = int(result.get("duration_sec", 0) * 1000)
           self.store.insert_ml_event("retrain", "scheduler", "local", latency_ms)
           self.store.commit()
       except Exception:
           logger.warning("scheduler: failed to log retrain event")
   ```

7. Verify: `grep -n "sqlite3" src/sigil_ml/training/scheduler.py` returns no results.

---

### Subtask T016 -- Update `app.py` to wire DataStore into routes, trainer, and scheduler (interim)

- **Purpose**: Make `app.py` create a DataStore and pass it to all consumers. This is interim wiring that WP05 will finalize.
- **Files**: `src/sigil_ml/app.py`
- **Parallel?**: No -- depends on T011-T015 completing the signature changes.

**Steps**:
1. Add import:
   ```python
   from sigil_ml.store import create_store
   ```

2. Add a `store` attribute to `AppState`:
   ```python
   class AppState:
       def __init__(self) -> None:
           self.store: DataStore | None = None  # Added
           self.stuck: StuckPredictor | None = None
           ...
   ```

3. Update `startup_event()` in `create_app()`:

   **Before** (lines 63-84):
   ```python
   async def startup_event() -> None:
       db = config.db_path()
       try:
           ensure_ml_tables(db)
       except Exception:
           ...
       state.load_models()
       state.poller = EventPoller(db_path=db, models={...})
       asyncio.create_task(state.poller.run())
       scheduler = TrainingScheduler(db, reload_callback=state.reload_models_into_poller)
   ```

   **After**:
   ```python
   async def startup_event() -> None:
       store = create_store()
       state.store = store
       try:
           store.ensure_tables()
       except Exception:
           ...
       state.load_models()
       state.poller = EventPoller(store=store, models={...})
       asyncio.create_task(state.poller.run())
       scheduler = TrainingScheduler(store, reload_callback=state.reload_models_into_poller)
   ```

4. Remove `from sigil_ml.schema import ensure_ml_tables` (will be fully cleaned up in WP05, but can be removed here since `store.ensure_tables()` replaces it).

5. Remove `from sigil_ml import config` if no longer used directly in app.py (it's used indirectly via `create_store()`).

---

## Risks & Mitigations

- **Risk**: Feature extraction functions have a wide call surface -- poller, routes, and training all call them. Signature change from `(db_path, task_id)` to `(store, task_id)` must be consistent everywhere. **Mitigation**: The call site inventory above enumerates every usage. After refactoring, `grep -rn "db_path" src/sigil_ml/` should only appear in `store_sqlite.py` and `config.py`.
- **Risk**: `routes.py` closure over `state` means the `state.store` must be set before any request arrives. **Mitigation**: The store is created in `startup_event()` which runs before the first request.
- **Risk**: The `TrainRequest.db` field becomes obsolete. **Mitigation**: Keep the field but ignore it (deprecation path). Remove in a future cleanup.

## Review Guidance

- `grep -r "import sqlite3" src/sigil_ml/` must return ONLY `store_sqlite.py`.
- `grep -rn "config.db_path" src/sigil_ml/` should only appear in `config.py` and `store.py` (the factory).
- Verify every call to `extract_stuck_features` and `extract_duration_features` now passes a `DataStore` not a path.
- Verify `Trainer` and `TrainingScheduler` constructors take `DataStore` not `db_path`.
- Verify `AppState` has a `store` attribute and it's set in `startup_event`.
- Verify `/status` endpoint uses `state.store` not `config.db_path()`.

## Activity Log

- 2026-03-30T01:45:06Z -- system -- lane=planned -- Prompt generated via /spec-kitty.tasks
