---
work_package_id: WP03
title: Refactor Features, Routes & Training to Use DataStore
lane: planned
dependencies: [WP01, WP02]
subtasks:
- T014
- T015
- T016
- T017
- T018
- T019
- T020
- T021
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
- FR-006
- FR-007
- FR-011
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

## Implementation Command

This WP depends on WP01 and WP02. Create the worktree with:

```bash
spec-kitty implement WP03 --base WP02
```

---

## Objectives & Success Criteria

- Eliminate **all** direct `sqlite3` usage from `features.py`, `routes.py`, `training/trainer.py`, and `training/scheduler.py`.
- Change feature extraction function signatures from `(db_path, task_id)` to `(store: DataStore, task_id)`.
- Route handlers use the `DataStore` from `AppState` instead of opening their own SQLite connections.
- Trainer and TrainingScheduler accept `DataStore` instead of `db_path`.
- **Success**: `grep -r "import sqlite3" src/sigil_ml/` returns ONLY `store_sqlite.py`. All functionality works identically through the DataStore.

## Context & Constraints

- **Spec**: `kitty-specs/002-storage-abstraction/spec.md` -- FR-005, FR-006, FR-007, FR-011
- **Prerequisites**: WP01 (DataStore protocol + SqliteStore) and WP02 (poller refactored) are complete.
- **Current sqlite3 usage in these modules**:
  - `features.py`: `_query_task()` and `_query_events_for_task()` open their own connections.
  - `routes.py`: `/status` endpoint opens a direct connection. `/predict/*` and `/train` endpoints pass `config.db_path()` to feature extractors and Trainer.
  - `training/trainer.py`: Constructor takes `db_path`, opens connections for task queries.
  - `training/scheduler.py`: Constructor takes `db_path`, opens connections for count queries and audit logging.
- **Call graph** (who calls feature extraction):
  - `poller.py:_predict_and_write` calls `extract_stuck_features(db_path, task_id)` and `extract_duration_features(db_path, task_id)` -- WP02 temporarily bridged these with `config.db_path()`.
  - `routes.py:predict_stuck` calls `extract_stuck_features(config.db_path(), req.task_id)`.
  - `routes.py:predict_duration` calls `extract_duration_features(config.db_path(), req.task_id)`.
  - `training/trainer.py:_train_stuck` calls `extract_stuck_features(self.db_path, task_id)`.
  - `training/trainer.py:_train_duration` calls `extract_duration_features(self.db_path, task_id)`.

## Subtasks & Detailed Guidance

### Subtask T014 -- Refactor `features.py` query helpers

- **Purpose**: Replace `_query_task` and `_query_events_for_task` with DataStore method calls. These are the foundational query functions used by all feature extractors.
- **Steps**:
  1. Open `src/sigil_ml/features.py`.
  2. Add import: `from sigil_ml.store import DataStore`
  3. Replace `_query_task(db_path, task_id)`:

     **Before**:
     ```python
     def _query_task(db_path: str | Path, task_id: str) -> dict[str, Any] | None:
         conn = sqlite3.connect(str(db_path))
         conn.row_factory = sqlite3.Row
         try:
             cur = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
             row = cur.fetchone()
             return dict(row) if row else None
         finally:
             conn.close()
     ```

     **After**:
     ```python
     def _query_task(store: DataStore, task_id: str) -> dict[str, Any] | None:
         return store.get_task_by_id(task_id)
     ```

  4. Replace `_query_events_for_task(db_path, task_id, since)`:

     **Before**: Opens its own connection, looks up task, queries events in time window.

     **After**:
     ```python
     def _query_events_for_task(store: DataStore, task_id: str, since: int | None = None) -> list[dict[str, Any]]:
         return store.get_events_for_task(task_id, since)
     ```

  5. Remove `import sqlite3` from the module (verify no other usage remains).
  6. Remove `from pathlib import Path` if no longer needed.

- **Files**: `src/sigil_ml/features.py` (modify `_query_task`, `_query_events_for_task`, remove sqlite3 import)
- **Parallel?**: Yes -- this is independent of routes and training changes.
- **Notes**: These are internal helpers. Their callers (`extract_stuck_features`, `extract_duration_features`) are updated in T015.

### Subtask T015 -- Update feature extraction function signatures

- **Purpose**: Change the public API of feature extraction functions from `(db_path, task_id)` to `(store: DataStore, task_id)`.
- **Steps**:
  1. Update `extract_stuck_features`:

     **Before**: `def extract_stuck_features(db_path: str | Path, task_id: str) -> dict[str, float]:`
     **After**: `def extract_stuck_features(store: DataStore, task_id: str) -> dict[str, float]:`

     Update internal calls:
     ```python
     task = _query_task(store, task_id)  # was _query_task(db_path, task_id)
     events = _query_events_for_task(store, task_id)  # was _query_events_for_task(db_path, task_id)
     ```

  2. Update `extract_duration_features`:

     **Before**: `def extract_duration_features(db_path: str | Path, task_id: str) -> dict[str, float]:`
     **After**: `def extract_duration_features(store: DataStore, task_id: str) -> dict[str, float]:`

     Update internal calls similarly.

  3. **Update ALL callers** (critical -- every call site must change):

     | Caller | File | Current Call | New Call |
     |--------|------|-------------|----------|
     | `EventPoller._predict_and_write` | `poller.py` | `extract_stuck_features(config.db_path(), task_id)` | `extract_stuck_features(self.store, task_id)` |
     | `EventPoller._predict_and_write` | `poller.py` | `extract_duration_features(config.db_path(), task_id)` | `extract_duration_features(self.store, task_id)` |
     | `predict_stuck` route | `routes.py` | `extract_stuck_features(config.db_path(), req.task_id)` | `extract_stuck_features(state.store, req.task_id)` |
     | `predict_duration` route | `routes.py` | `extract_duration_features(config.db_path(), req.task_id)` | `extract_duration_features(state.store, req.task_id)` |
     | `Trainer._train_stuck` | `trainer.py` | `extract_stuck_features(self.db_path, task_id)` | `extract_stuck_features(self.store, task_id)` |
     | `Trainer._train_duration` | `trainer.py` | `extract_duration_features(self.db_path, task_id)` | `extract_duration_features(self.store, task_id)` |

  4. Remove the `config.db_path()` temporary bridge from `poller.py` (added in WP02 T010).

- **Files**: `src/sigil_ml/features.py`, `src/sigil_ml/poller.py`, `src/sigil_ml/routes.py`, `src/sigil_ml/training/trainer.py`
- **Parallel?**: Yes -- but coordinate with T016-T020 since they also touch routes.py and trainer.py.
- **Notes**: `extract_features_from_buffer` and `extract_activity_features` do NOT take `db_path` -- they work from in-memory data. No change needed for those. `extract_workflow_features` also takes in-memory data only.

### Subtask T016 -- Refactor `/status` endpoint to use DataStore

- **Purpose**: The `/status` endpoint currently opens its own `sqlite3` connection to read cursor and predictions. Replace with DataStore calls.
- **Steps**:
  1. Open `src/sigil_ml/routes.py`.
  2. Locate the `status()` route handler.
  3. Replace the entire SQLite block:

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
                     "WHERE expires_at IS NULL OR expires_at > ? ORDER BY created_at DESC",
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
         if state.store is None:
             return {"cursor": None, "latest_predictions": [], "poller_running": False}
         try:
             cursor_info = state.store.get_cursor_status()
             preds = state.store.get_active_predictions()
             return {
                 "cursor": cursor_info,
                 "latest_predictions": preds,
                 "poller_running": state.poller is not None and state.poller._running,
             }
         except Exception:
             return {"cursor": None, "latest_predictions": [], "poller_running": False}
     ```

- **Files**: `src/sigil_ml/routes.py` (modify `status` handler)
- **Parallel?**: Yes -- independent of features.py and training changes.
- **Notes**: The error handling changes from catching `sqlite3.OperationalError` to a broader `Exception` since the DataStore may raise different exceptions.

### Subtask T017 -- Refactor `/predict/*` endpoints for DataStore

- **Purpose**: The prediction endpoints pass `config.db_path()` to feature extraction functions. After T015 changes their signatures, the routes must pass `state.store` instead.
- **Steps**:
  1. In `predict_stuck`:

     **Before**: `features = extract_stuck_features(config.db_path(), req.task_id)`
     **After**: `features = extract_stuck_features(state.store, req.task_id)`

  2. In `predict_duration`:

     **Before**: `features = extract_duration_features(config.db_path(), req.task_id)`
     **After**: `features = extract_duration_features(state.store, req.task_id)`

  3. Verify `predict_suggest` and `predict_quality` -- these do NOT use feature extraction from the database (they use in-memory data from the poller buffer or request body). No changes needed.

  4. Remove `from sigil_ml import config` if it is no longer used in `routes.py` after all changes.

- **Files**: `src/sigil_ml/routes.py` (modify `predict_stuck`, `predict_duration`)
- **Parallel?**: Yes -- but coordinate with T015 (signature changes) and T016 (same file).
- **Notes**: Only two prediction endpoints actually query the database. The others (`suggest`, `quality`) take features from the request body.

### Subtask T018 -- Refactor `/train` endpoint and `_run_training`

- **Purpose**: The `/train` endpoint creates a `Trainer(db_path)`. After T019 changes Trainer's constructor, this must pass the DataStore instead.
- **Steps**:
  1. Update `_run_training`:

     **Before**:
     ```python
     def _run_training(state: AppState, db_path: str) -> None:
         try:
             state.training_in_progress = True
             trainer = Trainer(db_path)
             result = trainer.train_all()
             ...
     ```

     **After**:
     ```python
     def _run_training(state: AppState) -> None:
         try:
             state.training_in_progress = True
             trainer = Trainer(state.store)
             result = trainer.train_all()
             ...
     ```

  2. Update the `/train` endpoint handler:

     **Before**:
     ```python
     db = req.db or str(config.db_path())
     background_tasks.add_task(_run_training, state, db)
     ```

     **After**:
     ```python
     background_tasks.add_task(_run_training, state)
     ```

  3. The `req.db` override field in `TrainRequest` becomes obsolete (or could be kept for testing but ignored). Consider deprecating but not removing to avoid API breakage.

  4. Remove `import sqlite3` from `routes.py` (verify no other usage remains).
  5. Remove `import time` if no longer used (was used for `int(time.time() * 1000)` in `/status`).

- **Files**: `src/sigil_ml/routes.py` (modify `_run_training`, `train` handler, clean up imports)
- **Parallel?**: Yes -- but depends on T019 (Trainer refactor) being done first or simultaneously.
- **Notes**: The `req.db` override was useful for testing with a specific SQLite path. With the DataStore, testing should use a mock store instead.

### Subtask T019 -- Refactor Trainer to accept DataStore

- **Purpose**: The `Trainer` class currently takes `db_path` and opens its own SQLite connections. Replace with DataStore.
- **Steps**:
  1. Open `src/sigil_ml/training/trainer.py`.
  2. Change the constructor:

     **Before**:
     ```python
     def __init__(self, db_path: str | Path) -> None:
         self.db_path = Path(db_path)
     ```

     **After**:
     ```python
     def __init__(self, store: DataStore) -> None:
         self.store = store
     ```

  3. Refactor `_train_stuck`:

     **Before**: Opens `sqlite3.connect(str(self.db_path))`, queries `SELECT id FROM tasks WHERE completed_at IS NOT NULL`, then calls `extract_stuck_features(self.db_path, task_id)`.

     **After**:
     ```python
     def _train_stuck(self) -> int:
         task_ids = self.store.get_completed_task_ids()

         if len(task_ids) < 10:
             logger.info("Not enough completed tasks for stuck training (%d)", len(task_ids))
             X, y = generate_stuck_data(500)
             predictor = StuckPredictor()
             predictor.train(X, y)
             return 500

         X_list = []
         y_list = []
         for task_id in task_ids:
             features = extract_stuck_features(self.store, task_id)
             x = [features.get(f, 0.0) for f in STUCK_FEATURES]
             X_list.append(x)
             stuck = features["test_failure_count"] > 3 and features["time_in_phase_sec"] > 600
             y_list.append(1.0 if stuck else 0.0)

         X = np.array(X_list)
         y = np.array(y_list)
         predictor = StuckPredictor()
         predictor.train(X, y)
         return len(X)
     ```

  4. Refactor `_train_duration` similarly:
     - Replace `sqlite3.connect` query with `self.store.get_completed_tasks_with_timing()`.
     - Replace `extract_duration_features(self.db_path, task_id)` with `extract_duration_features(self.store, task_id)`.

  5. Remove `import sqlite3` and `from pathlib import Path` from the module.
  6. Add `from sigil_ml.store import DataStore`.

- **Files**: `src/sigil_ml/training/trainer.py` (full refactor)
- **Parallel?**: Yes -- independent of routes and features refactoring.
- **Notes**: The `db_path.exists()` check at the start of `_train_stuck` and `_train_duration` was SQLite-specific. With DataStore, skip this check -- the store handles connection errors internally.

### Subtask T020 -- Refactor TrainingScheduler to accept DataStore

- **Purpose**: The `TrainingScheduler` currently takes `db_path` and opens its own SQLite connections for counting tasks and logging audit events.
- **Steps**:
  1. Open `src/sigil_ml/training/scheduler.py`.
  2. Change the constructor:

     **Before**:
     ```python
     def __init__(self, db_path: Path, reload_callback) -> None:
         self.db_path = db_path
         ...
     ```

     **After**:
     ```python
     def __init__(self, store: DataStore, reload_callback) -> None:
         self.store = store
         ...
     ```

  3. Refactor `_count_completed`:

     **Before**: Opens `sqlite3.connect`, executes `SELECT COUNT(*)`.

     **After**:
     ```python
     def _count_completed(self) -> int:
         try:
             return self.store.count_completed_tasks()
         except Exception:
             return 0
     ```

  4. Refactor `_log_retrain`:

     **Before**: Opens `sqlite3.connect`, inserts into `ml_events`.

     **After**:
     ```python
     def _log_retrain(self, result: dict) -> None:
         try:
             latency_ms = int(result.get("duration_sec", 0) * 1000)
             self.store.insert_ml_event("retrain", "scheduler", "local", latency_ms)
         except Exception:
             logger.warning("scheduler: failed to log retrain event")
     ```

  5. Refactor `check_and_retrain` -- the `Trainer` instantiation:

     **Before**: `result = Trainer(self.db_path).train_all()`
     **After**: `result = Trainer(self.store).train_all()`

  6. Remove `import sqlite3` and update imports.

- **Files**: `src/sigil_ml/training/scheduler.py` (full refactor)
- **Parallel?**: Yes -- independent of features and routes.
- **Notes**: The `sqlite3.OperationalError` catches in `_count_completed` and `_log_retrain` become generic `Exception` catches.

### Subtask T021 -- Update app.py wiring for features, routes, and training

- **Purpose**: Complete the wiring so that all components receive the DataStore from app startup.
- **Steps**:
  1. Open `src/sigil_ml/app.py`.
  2. In `startup_event()`, update the `TrainingScheduler` instantiation:

     **Before**: `scheduler = TrainingScheduler(db, reload_callback=state.reload_models_into_poller)`
     **After**: `scheduler = TrainingScheduler(store, reload_callback=state.reload_models_into_poller)`

  3. Verify the `store` is already on `state` (set in WP02 T013): `state.store = store`.
  4. Verify routes can access `state.store` -- the `register_routes(application, state)` call already passes `state`, and routes access `state.store`.
  5. Remove `db = config.db_path()` from startup if it is no longer used directly.
  6. Remove `from sigil_ml.schema import ensure_ml_tables` if `store.ensure_tables()` is being called instead (check if WP02 already did this).
  7. Verify the complete startup flow:
     ```python
     store = create_store()
     store.ensure_tables()
     state.store = store
     state.load_models()
     state.poller = EventPoller(store=store, models={...})
     scheduler = TrainingScheduler(store, reload_callback=...)
     ```

- **Files**: `src/sigil_ml/app.py` (finalize wiring)
- **Parallel?**: No -- should be done after T014-T020 are complete.
- **Notes**: After this subtask, do a final check: `grep -r "import sqlite3" src/sigil_ml/` should return only `store_sqlite.py`. If `schema.py` still imports it, that is addressed in WP05.

## Risks & Mitigations

- **Risk**: Wide call surface for feature extraction signature changes -- easy to miss a call site. **Mitigation**: The call site table in T015 is exhaustive. After changing signatures, run `python -c "import sigil_ml.app"` to catch import-time errors.
- **Risk**: Routes rely on `state.store` which is set during startup. If routes are called before startup completes, `state.store` is None. **Mitigation**: Add None checks in route handlers (already present in `/status` refactor).
- **Risk**: Trainer's `db_path.exists()` check is SQLite-specific and can't be replicated for DataStore. **Mitigation**: Remove the check; the DataStore handles connection errors internally and returns empty results gracefully.

## Review Guidance

- **Primary check**: `grep -r "import sqlite3" src/sigil_ml/` returns only `store_sqlite.py` (and possibly `schema.py` which is retired in WP05).
- Verify every feature extraction call site passes `store` instead of `db_path`.
- Verify `Trainer` and `TrainingScheduler` constructors accept `DataStore`.
- Verify `/status` endpoint uses DataStore methods.
- Run the full test suite -- no regressions.

## Activity Log

- 2026-03-29T16:29:57Z -- system -- lane=planned -- Prompt created.
