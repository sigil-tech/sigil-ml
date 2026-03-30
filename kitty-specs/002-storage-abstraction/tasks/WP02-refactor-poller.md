---
work_package_id: "WP02"
title: "Refactor Poller to Use DataStore"
lane: "planned"
dependencies: ["WP01"]
subtasks:
  - "T006"
  - "T007"
  - "T008"
  - "T009"
  - "T010"
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
  - "FR-011"
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

## Objectives & Success Criteria

- Refactor `EventPoller` to accept a `DataStore` instance instead of `db_path: Path`.
- Eliminate ALL direct `sqlite3` usage from `src/sigil_ml/poller.py`.
- After this WP, `poller.py` should have zero `import sqlite3` and zero `sqlite3.` references.
- The poller must call only `DataStore` methods for all data access.
- All existing poller behavior (poll -> classify -> predict -> write -> commit) remains identical.

**Success criteria**:
- FR-005: The EventPoller uses the DataStore protocol instead of direct SQLite access.
- FR-011: The poller is backend-agnostic -- same code works with SQLite or Postgres store.

**Implementation command**: `spec-kitty implement WP02 --base WP01`

## Context & Constraints

- **Prerequisite**: WP01 must be complete (DataStore protocol and SqliteStore exist).
- **Spec**: `kitty-specs/002-storage-abstraction/spec.md` -- User Story 3 (Poller uses DataStore interface).
- **Current state**: `poller.py` is 300 lines. It creates connections via `_connect()`, passes `conn` to helper methods, and commits at the end of each poll cycle.
- **Key pattern to preserve**: The poller currently uses a single connection per poll cycle with a final `conn.commit()`. The DataStore's `commit()` method replaces this.

### Current Architecture (before refactor)

```
EventPoller.__init__(db_path, models)
  |
  v
_poll_once()
  conn = _connect()                    # <-- REMOVE
  cursor = conn.execute(SELECT cursor) # <-- becomes store.get_cursor()
  rows = conn.execute(SELECT events)   # <-- becomes store.get_events_since()
  conn.execute(UPDATE cursor)          # <-- becomes store.update_cursor()
  _predict_and_write(conn)             # <-- conn param removed
  conn.commit()                        # <-- becomes store.commit()
  conn.close()                         # <-- REMOVE
```

### Target Architecture (after refactor)

```
EventPoller.__init__(store: DataStore, models)
  |
  v
_poll_once()
  cursor = store.get_cursor()
  events = store.get_events_since(cursor)
  store.update_cursor(max_id)
  _predict_and_write()
  store.commit()
```

---

## Subtasks & Detailed Guidance

### Subtask T006 -- Change `EventPoller.__init__` to accept `DataStore`; remove `_connect`

- **Purpose**: Replace the `db_path` dependency with a `DataStore` dependency. Remove the `_connect()` method that creates raw SQLite connections.
- **Files**: `src/sigil_ml/poller.py`
- **Parallel?**: No -- must be done first; T007-T009 depend on this.

**Steps**:
1. Change the constructor signature:

   **Before** (line 33):
   ```python
   def __init__(self, db_path: Path, models: dict) -> None:
       self.db_path = db_path
   ```

   **After**:
   ```python
   def __init__(self, store: DataStore, models: dict) -> None:
       self.store = store
   ```

2. Add the import at the top of the file:
   ```python
   from sigil_ml.store import DataStore
   ```

3. Remove the `_connect` method (lines 295-298):
   ```python
   # DELETE THIS ENTIRE METHOD:
   def _connect(self) -> sqlite3.Connection:
       conn = sqlite3.connect(str(self.db_path), timeout=5.0)
       conn.execute("PRAGMA journal_mode=WAL")
       conn.execute("PRAGMA busy_timeout=5000")
       return conn
   ```

4. Remove `from pathlib import Path` import if no longer used.

**Notes**:
- The `db_path` attribute is referenced in the `run()` method's log message (line 48). Change it to log the store type instead:
  ```python
  logger.info("poller: started with %s", type(self.store).__name__)
  ```

---

### Subtask T007 -- Refactor `_poll_once` to use DataStore methods

- **Purpose**: Replace the inline SQL in `_poll_once()` with DataStore method calls.
- **Files**: `src/sigil_ml/poller.py`
- **Parallel?**: Yes -- can proceed in parallel with T008, T009 after T006 is done.

**Steps**:
1. Replace the cursor read (line 65):

   **Before**:
   ```python
   cursor_id = conn.execute("SELECT last_event_id FROM ml_cursor WHERE id = 1").fetchone()
   since = cursor_id[0] if cursor_id else 0
   ```

   **After**:
   ```python
   since = self.store.get_cursor()
   ```

2. Replace the events query (lines 68-71):

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

3. Adjust the event processing loop. Currently (lines 76-89) it processes `rows` as tuples via `dict(zip(...))`. After refactoring, `get_events_since` returns list of dicts already, so:

   **Before**:
   ```python
   events = []
   for row in rows:
       e = dict(zip(["id", "kind", "source", "payload", "ts"], row))
       if isinstance(e.get("payload"), str):
           ...
   ```

   **After**:
   ```python
   events = []
   for e in rows:
       if isinstance(e.get("payload"), str):
           try:
               e["payload"] = json.loads(e["payload"])
           except (json.JSONDecodeError, TypeError):
               pass
       classification = self.activity.classify(e)
       e["_category"] = classification["category"]
       e["_category_confidence"] = classification["confidence"]
       events.append(e)
   ```

4. Replace the cursor update (lines 97-99):

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

5. Replace `conn.commit()` (line 108) with `self.store.commit()`.

6. Remove `conn = self._connect()` (line 63) and the `try/finally conn.close()` wrapper (lines 64, 109-110). The DataStore manages its own connection lifecycle.

7. The full `_poll_once()` method structure becomes:
   ```python
   def _poll_once(self) -> None:
       since = self.store.get_cursor()
       rows = self.store.get_events_since(since, limit=100)
       if not rows:
           return

       events = []
       for e in rows:
           # ... classify events (unchanged) ...
           events.append(e)

       self._buffer.extend(events)
       self._buffer = self._buffer[-200:]
       self._since_last_predict += len(events)

       max_id = max(e["id"] for e in events)
       self.store.update_cursor(max_id)

       elapsed = time.time() - self._last_predict_time
       if self._since_last_predict >= PREDICT_EVERY_N_EVENTS and elapsed >= PREDICT_MIN_INTERVAL_SEC:
           self._predict_and_write()
           self._since_last_predict = 0
           self._last_predict_time = time.time()

       self.store.commit()
   ```

---

### Subtask T008 -- Refactor `_predict_and_write` to use DataStore methods

- **Purpose**: Remove the `conn` parameter from `_predict_and_write` and replace all SQL operations with DataStore calls.
- **Files**: `src/sigil_ml/poller.py`
- **Parallel?**: Yes -- can proceed in parallel with T007, T009.

**Steps**:
1. Change the method signature:

   **Before** (line 133):
   ```python
   def _predict_and_write(self, conn: sqlite3.Connection) -> None:
   ```

   **After**:
   ```python
   def _predict_and_write(self) -> None:
   ```

2. Replace `self._active_task_id(conn)` (line 135) with `self.store.get_active_task()`.

3. Replace `extract_stuck_features(self.db_path, task_id)` (line 140) with `extract_stuck_features(self.store, task_id)`.
   - **Important**: This signature change happens in WP03. For now, the poller must pass `self.store` instead of `self.db_path`. If WP03 is not yet complete, use `self.store.db_path` as a temporary bridge. Prefer passing the store directly.

4. Replace `self._write(conn, ...)` calls (lines 146, 150, 155, 170, 175) with `self._write(...)` (no conn param).

5. Replace `extract_duration_features(self.db_path, task_id)` (line 160) with `extract_duration_features(self.store, task_id)` (same WP03 note as step 3).

6. Replace the session info call. Currently `self._session_info(conn, task_id)` (line 153). After refactor: `self._session_info(task_id)`.

7. Replace the quality features call. Currently `self._quality_features(conn)` (line 172). After refactor: `self._quality_features()`.

8. Replace the inline audit log write (lines 179-183):

   **Before**:
   ```python
   conn.execute(
       "INSERT INTO ml_events (kind, endpoint, routing, latency_ms, ts) "
       "VALUES ('prediction', 'poller', 'local', ?, ?)",
       (latency_ms, int(time.time() * 1000)),
   )
   ```

   **After**:
   ```python
   self.store.insert_ml_event("prediction", "poller", "local", latency_ms)
   ```

9. Remove the `_active_task_id` method (lines 249-252) -- it's replaced by `self.store.get_active_task()`.

10. Refactor the `_write` method (lines 234-247):

    **Before**:
    ```python
    def _write(self, conn: sqlite3.Connection, model: str, result: dict,
               confidence: float, ttl_sec: int | None) -> None:
        ...
        conn.execute("INSERT INTO ml_predictions ...")
    ```

    **After**:
    ```python
    def _write(self, model: str, result: dict, confidence: float,
               ttl_sec: int | None) -> None:
        self.store.insert_prediction(model, result, confidence, ttl_sec)
    ```

    Or simply inline the `store.insert_prediction()` call at each call site and remove `_write` entirely.

---

### Subtask T009 -- Refactor `_session_info` and `_quality_features` to use DataStore

- **Purpose**: Replace the remaining SQL operations in the poller's helper methods.
- **Files**: `src/sigil_ml/poller.py`
- **Parallel?**: Yes -- can proceed in parallel with T007, T008.

**Steps**:
1. Refactor `_session_info` (lines 211-232):

   **Before**:
   ```python
   def _session_info(self, conn: sqlite3.Connection, task_id: str | None) -> dict:
       ...
       if task_id:
           row = conn.execute(
               "SELECT started_at, phase, test_fails FROM tasks WHERE id = ?",
               (task_id,),
           ).fetchone()
   ```

   **After**:
   ```python
   def _session_info(self, task_id: str | None) -> dict:
       session_elapsed_min = 0.0
       task_phase = None
       test_failures = 0

       if task_id:
           info = self.store.get_session_info(task_id)
           if info:
               started_at = info["started_at"] or 0
               session_elapsed_min = (time.time() * 1000 - started_at) / 60000.0
               task_phase = info["phase"]
               test_failures = info["test_fails"] or 0

       return {
           "session_elapsed_min": max(session_elapsed_min, 0.0),
           "task_phase": task_phase,
           "test_failures": test_failures,
       }
   ```

2. Refactor `_quality_features` (lines 255-293):

   **Before**:
   ```python
   def _quality_features(self, conn: sqlite3.Connection) -> dict:
       ...
       row = conn.execute(
           "SELECT test_runs, test_fails, commit_count FROM tasks "
           "WHERE completed_at IS NOT NULL ORDER BY completed_at DESC LIMIT 1"
       ).fetchone()
   ```

   **After**:
   ```python
   def _quality_features(self) -> dict:
       now_ms = int(time.time() * 1000)
       window_start = now_ms - QUALITY_WINDOW_SEC * 1000
       window = [e for e in self._buffer if e.get("ts", 0) >= window_start]

       # ... edit_events, files, edit_focus logic unchanged ...

       stats = self.store.get_quality_task_stats()

       if stats and stats["test_runs"]:
           test_pass_rate = 1.0 - (stats["test_fails"] / max(stats["test_runs"], 1))
           baseline_commits = max(stats["commit_count"], 1)
       else:
           test_pass_rate = 0.7
           baseline_commits = 1

       # ... rest of return dict unchanged ...
   ```

   Note: The buffer processing logic (edit events, files, commit events, terminal events) is pure in-memory work and does NOT touch the database. Only the `conn.execute(SELECT ... FROM tasks ...)` call needs to be replaced.

---

### Subtask T010 -- Remove all `sqlite3` imports from `poller.py`

- **Purpose**: Final cleanup -- verify and remove all `sqlite3` references.
- **Files**: `src/sigil_ml/poller.py`
- **Parallel?**: No -- must be done after T006-T009 are complete.

**Steps**:
1. Remove `import sqlite3` (line 10).

2. Update the error handling in `run()` (line 52):

   **Before**:
   ```python
   except sqlite3.OperationalError as e:
       logger.debug("poller: sqlite error (will retry): %s", e)
   ```

   **After**:
   ```python
   except Exception as e:
       logger.debug("poller: store error (will retry): %s", e)
   ```

   Alternatively, if a `StoreError` base exception is defined in the DataStore module, catch that specifically. For now, catching `Exception` and logging at debug level is safe (matches the current "retry silently" pattern).

3. Remove `from pathlib import Path` if no longer used.

4. Verify: Run `grep -n "sqlite3" src/sigil_ml/poller.py` -- should return no results.

5. Verify: The only imports from `sigil_ml.store` should be `DataStore`.

---

## Risks & Mitigations

- **Risk**: The poller's transactional pattern (poll + predict + cursor update in one commit) must be preserved. If the DataStore commits automatically per operation, the atomicity guarantee breaks. **Mitigation**: The `SqliteStore` uses a persistent connection within a poll cycle and only commits when `commit()` is called.
- **Risk**: The `features.extract_stuck_features` and `extract_duration_features` still take `db_path` until WP03 refactors them. **Mitigation**: During WP02, pass `self.store.db_path` temporarily to maintain compatibility. Document this as a known tech debt resolved in WP03.
- **Risk**: `sqlite3.OperationalError` is caught in `run()` to handle missing database. Generalizing to `Exception` might mask real bugs. **Mitigation**: Log at debug level (matching current behavior) and keep the "will retry" pattern.

## Review Guidance

- Verify `poller.py` has ZERO `sqlite3` references after this WP.
- Verify the `_poll_once` method still follows: get_cursor -> get_events -> classify -> update_cursor -> predict_and_write -> commit.
- Verify `_predict_and_write` no longer takes a `conn` parameter.
- Verify `_session_info` and `_quality_features` no longer take a `conn` parameter.
- Verify the `_connect` and `_active_task_id` methods are removed.
- Check that `app.py` still starts the poller correctly (interim wiring -- full wiring in WP05).

## Activity Log

- 2026-03-30T01:45:06Z -- system -- lane=planned -- Prompt generated via /spec-kitty.tasks
