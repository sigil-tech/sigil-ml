---
work_package_id: WP01
title: DataStore Protocol & SQLite Implementation
lane: planned
dependencies: []
subtasks:
- T001
- T002
- T003
- T004
- T005
- T006
- T007
phase: Phase 1 - Foundation
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
- FR-001
- FR-002
- FR-003
- FR-008
- FR-010
- FR-011
---

# Work Package Prompt: WP01 -- DataStore Protocol & SQLite Implementation

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

- Define a `DataStore` protocol that covers **every** SQL operation currently performed directly against SQLite across the entire codebase.
- Implement `SqliteStore` that preserves identical behavior to today's inline `sqlite3` calls, including WAL mode and busy_timeout.
- Provide a `create_store()` factory function that selects the backend based on configuration.
- **Success**: `SqliteStore` can be instantiated and all its methods return the same data shapes as current direct queries. No behavioral change for end users.

## Context & Constraints

- **Spec**: `kitty-specs/002-storage-abstraction/spec.md` -- FR-001, FR-002, FR-003, FR-008, FR-010, FR-011
- **CLAUDE.md invariants**: Every SQLite connection must set `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=5000`. Model names must be `"stuck"`, `"suggest"`, `"duration"`, `"quality"`. Python only writes to `ml_predictions`, `ml_events`, `ml_cursor`.
- **Dependency constraint**: `sigil-ml` allows only `scikit-learn`, `numpy`, `fastapi`, `uvicorn`, `joblib`. No new heavy dependencies in this WP.
- **Existing SQL operations audit** (exhaustive list of every `sqlite3` call in the codebase):

  | Module | Operation | SQL |
  |--------|-----------|-----|
  | `schema.py` | Ensure ml_cursor table | `CREATE TABLE IF NOT EXISTS ml_cursor ...` + `INSERT OR IGNORE` |
  | `poller.py` `_connect` | Open connection | WAL + busy_timeout pragmas |
  | `poller.py` `_poll_once` | Read cursor | `SELECT last_event_id FROM ml_cursor WHERE id = 1` |
  | `poller.py` `_poll_once` | Get events since cursor | `SELECT id, kind, source, payload, ts FROM events WHERE id > ? ORDER BY id ASC LIMIT 100` |
  | `poller.py` `_poll_once` | Update cursor | `UPDATE ml_cursor SET last_event_id = ?, updated_at = ? WHERE id = 1` |
  | `poller.py` `_predict_and_write` | Get active task | `SELECT id FROM tasks WHERE phase != 'idle' AND completed_at IS NULL ORDER BY last_active DESC LIMIT 1` |
  | `poller.py` `_predict_and_write` | Write prediction | `INSERT INTO ml_predictions (model, result, confidence, created_at, expires_at) VALUES (?, ?, ?, ?, ?)` |
  | `poller.py` `_predict_and_write` | Write audit event | `INSERT INTO ml_events (kind, endpoint, routing, latency_ms, ts) VALUES (...)` |
  | `poller.py` `_session_info` | Get task session info | `SELECT started_at, phase, test_fails FROM tasks WHERE id = ?` |
  | `poller.py` `_quality_features` | Get completed task stats | `SELECT test_runs, test_fails, commit_count FROM tasks WHERE completed_at IS NOT NULL ORDER BY completed_at DESC LIMIT 1` |
  | `features.py` `_query_task` | Get task by ID | `SELECT * FROM tasks WHERE id = ?` |
  | `features.py` `_query_events_for_task` | Get events in time window | `SELECT * FROM events WHERE ts >= ? AND ts <= ? ORDER BY ts` |
  | `routes.py` `/status` | Read cursor | `SELECT last_event_id, updated_at FROM ml_cursor WHERE id = 1` |
  | `routes.py` `/status` | Read active predictions | `SELECT model, confidence, created_at FROM ml_predictions WHERE expires_at IS NULL OR expires_at > ? ORDER BY created_at DESC` |
  | `trainer.py` `_train_stuck` | Get completed task IDs | `SELECT id FROM tasks WHERE completed_at IS NOT NULL` |
  | `trainer.py` `_train_duration` | Get completed tasks with timing | `SELECT id, started_at, completed_at FROM tasks WHERE completed_at IS NOT NULL AND started_at IS NOT NULL` |
  | `scheduler.py` `_count_completed` | Count completed tasks | `SELECT COUNT(*) FROM tasks WHERE completed_at IS NOT NULL` |
  | `scheduler.py` `_log_retrain` | Write audit event | Same as poller audit insert |

## Subtasks & Detailed Guidance

### Subtask T001 -- Define the DataStore protocol in `src/sigil_ml/store.py`

- **Purpose**: Establish the single interface contract that all consumers depend on. This is the central abstraction (FR-001, FR-002).
- **Steps**:
  1. Create `src/sigil_ml/store.py`.
  2. Define a `DataStore` class using `typing.Protocol` (Python 3.8+ `typing_extensions` or 3.12 `typing`).
  3. Include the following method signatures derived from the SQL audit above:

  ```python
  from typing import Protocol, Any

  class DataStore(Protocol):
      """Protocol for all data access operations in sigil-ml."""

      def ensure_tables(self) -> None:
          """Create Python-owned tables if they don't exist (ml_cursor)."""
          ...

      def get_cursor(self) -> int:
          """Return the last_event_id from ml_cursor. Returns 0 if no cursor."""
          ...

      def update_cursor(self, last_event_id: int) -> None:
          """Update the poll cursor to the given event ID."""
          ...

      def get_events_since(self, since_id: int, limit: int = 100) -> list[dict[str, Any]]:
          """Return events with id > since_id, ordered by id ASC, up to limit.

          Each dict has keys: id, kind, source, payload (parsed JSON), ts.
          """
          ...

      def get_active_task(self) -> str | None:
          """Return the ID of the active task, or None."""
          ...

      def get_task_by_id(self, task_id: str) -> dict[str, Any] | None:
          """Return a full task row as a dict, or None if not found."""
          ...

      def get_task_session_info(self, task_id: str) -> dict[str, Any] | None:
          """Return started_at, phase, test_fails for a task. None if not found."""
          ...

      def get_events_for_task(self, task_id: str, since: int | None = None) -> list[dict[str, Any]]:
          """Return events within a task's time window (started_at..completed_at/last_active)."""
          ...

      def get_completed_task_ids(self) -> list[str]:
          """Return IDs of all completed tasks."""
          ...

      def get_completed_tasks_with_timing(self) -> list[dict[str, Any]]:
          """Return id, started_at, completed_at for completed tasks with both timestamps."""
          ...

      def count_completed_tasks(self) -> int:
          """Return the count of completed tasks."""
          ...

      def get_quality_task_stats(self) -> dict[str, Any] | None:
          """Return test_runs, test_fails, commit_count from most recently completed task."""
          ...

      def get_active_predictions(self) -> list[dict[str, Any]]:
          """Return non-expired predictions ordered by created_at DESC."""
          ...

      def get_cursor_status(self) -> dict[str, Any] | None:
          """Return last_event_id and updated_at from ml_cursor. For /status endpoint."""
          ...

      def insert_prediction(
          self, model: str, result: dict, confidence: float, ttl_sec: int | None
      ) -> None:
          """Write a prediction row to ml_predictions."""
          ...

      def insert_ml_event(
          self, kind: str, endpoint: str, routing: str, latency_ms: int
      ) -> None:
          """Write an audit row to ml_events."""
          ...
  ```

  4. Also export a `create_store` function signature (implemented in T007).

- **Files**: `src/sigil_ml/store.py` (new file, ~100 lines)
- **Parallel?**: No -- this must be done first as all other subtasks reference the protocol.
- **Notes**: Use `typing.Protocol` for structural subtyping. Consumers type-hint against `DataStore` without inheritance. Include docstrings on every method explaining the expected return shape.

### Subtask T002 -- Implement SqliteStore cursor operations

- **Purpose**: Extract cursor read/update logic from `poller.py` into `SqliteStore`.
- **Steps**:
  1. Create `src/sigil_ml/store_sqlite.py`.
  2. Implement `SqliteStore.__init__(self, db_path: Path)` that stores the path.
  3. Implement a private `_connect(self) -> sqlite3.Connection` method that opens a connection with WAL mode and busy_timeout=5000 (matching `poller.py:_connect` exactly).
  4. Implement `get_cursor(self) -> int`:
     ```python
     conn = self._connect()
     try:
         row = conn.execute("SELECT last_event_id FROM ml_cursor WHERE id = 1").fetchone()
         return row[0] if row else 0
     finally:
         conn.close()
     ```
  5. Implement `update_cursor(self, last_event_id: int) -> None`:
     ```python
     conn = self._connect()
     try:
         conn.execute(
             "UPDATE ml_cursor SET last_event_id = ?, updated_at = ? WHERE id = 1",
             (last_event_id, int(time.time() * 1000)),
         )
         conn.commit()
     finally:
         conn.close()
     ```
  6. Implement `get_cursor_status(self) -> dict | None` for the `/status` endpoint.

- **Files**: `src/sigil_ml/store_sqlite.py` (new file, begin building ~200 lines)
- **Parallel?**: Yes -- can proceed once T001 is complete.
- **Notes**: The `_connect` helper is reused by all subsequent methods. It must exactly replicate the WAL and busy_timeout pragmas.

### Subtask T003 -- Implement SqliteStore event querying operations

- **Purpose**: Extract event query logic from `poller.py` and `features.py` into `SqliteStore`.
- **Steps**:
  1. Implement `get_events_since(self, since_id: int, limit: int = 100) -> list[dict]`:
     - Query: `SELECT id, kind, source, payload, ts FROM events WHERE id > ? ORDER BY id ASC LIMIT ?`
     - Parse JSON payload for each row (matching `poller.py:_poll_once` logic).
     - Return list of dicts with keys: `id`, `kind`, `source`, `payload` (parsed), `ts`.
  2. Implement `get_events_for_task(self, task_id: str, since: int | None = None) -> list[dict]`:
     - Look up the task to get `started_at` and `completed_at`/`last_active` (reuse `get_task_by_id`).
     - Query: `SELECT * FROM events WHERE ts >= ? AND ts <= ? ORDER BY ts`
     - Parse JSON payload for each row (matching `features.py:_query_events_for_task`).

- **Files**: `src/sigil_ml/store_sqlite.py` (add methods)
- **Parallel?**: Yes -- can proceed once T001 is complete.
- **Notes**: Payload parsing must handle both JSON strings and already-parsed dicts, matching current behavior.

### Subtask T004 -- Implement SqliteStore task querying operations

- **Purpose**: Extract task query logic from `poller.py`, `features.py`, `trainer.py`, and `scheduler.py`.
- **Steps**:
  1. Implement `get_active_task(self) -> str | None`:
     - Query: `SELECT id FROM tasks WHERE phase != 'idle' AND completed_at IS NULL ORDER BY last_active DESC LIMIT 1`
  2. Implement `get_task_by_id(self, task_id: str) -> dict | None`:
     - Query: `SELECT * FROM tasks WHERE id = ?` with `row_factory = sqlite3.Row`
     - Return `dict(row)` or `None`.
  3. Implement `get_task_session_info(self, task_id: str) -> dict | None`:
     - Query: `SELECT started_at, phase, test_fails FROM tasks WHERE id = ?`
     - Return dict with those keys, or `None`.
  4. Implement `get_completed_task_ids(self) -> list[str]`:
     - Query: `SELECT id FROM tasks WHERE completed_at IS NOT NULL`
  5. Implement `get_completed_tasks_with_timing(self) -> list[dict]`:
     - Query: `SELECT id, started_at, completed_at FROM tasks WHERE completed_at IS NOT NULL AND started_at IS NOT NULL`
  6. Implement `count_completed_tasks(self) -> int`:
     - Query: `SELECT COUNT(*) FROM tasks WHERE completed_at IS NOT NULL`
  7. Implement `get_quality_task_stats(self) -> dict | None`:
     - Query: `SELECT test_runs, test_fails, commit_count FROM tasks WHERE completed_at IS NOT NULL ORDER BY completed_at DESC LIMIT 1`

- **Files**: `src/sigil_ml/store_sqlite.py` (add methods)
- **Parallel?**: Yes -- can proceed once T001 is complete.
- **Notes**: Each method opens and closes its own connection. This matches the current per-call pattern in `features.py` and `trainer.py`.

### Subtask T005 -- Implement SqliteStore prediction and audit writing operations

- **Purpose**: Extract write operations from `poller.py` and `scheduler.py`.
- **Steps**:
  1. Implement `insert_prediction(self, model: str, result: dict, confidence: float, ttl_sec: int | None) -> None`:
     ```python
     now_ms = int(time.time() * 1000)
     expires_ms = (now_ms + ttl_sec * 1000) if ttl_sec else None
     conn = self._connect()
     try:
         conn.execute(
             "INSERT INTO ml_predictions (model, result, confidence, created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
             (model, json.dumps(result), round(confidence, 4), now_ms, expires_ms),
         )
         conn.commit()
     finally:
         conn.close()
     ```
  2. Implement `insert_ml_event(self, kind: str, endpoint: str, routing: str, latency_ms: int) -> None`:
     ```python
     conn = self._connect()
     try:
         conn.execute(
             "INSERT INTO ml_events (kind, endpoint, routing, latency_ms, ts) VALUES (?, ?, ?, ?, ?)",
             (kind, endpoint, routing, latency_ms, int(time.time() * 1000)),
         )
         conn.commit()
     finally:
         conn.close()
     ```
  3. Implement `get_active_predictions(self) -> list[dict]` for the `/status` endpoint:
     - Query: `SELECT model, confidence, created_at FROM ml_predictions WHERE expires_at IS NULL OR expires_at > ? ORDER BY created_at DESC`

- **Files**: `src/sigil_ml/store_sqlite.py` (add methods)
- **Parallel?**: Yes -- can proceed once T001 is complete.
- **Notes**: Model name values must exactly match Go's expectations: `"stuck"`, `"suggest"`, `"duration"`, `"quality"`. This is enforced by callers, not the store.

### Subtask T006 -- Implement SqliteStore schema bootstrap

- **Purpose**: Replace `schema.ensure_ml_tables()` with `SqliteStore.ensure_tables()`.
- **Steps**:
  1. Implement `ensure_tables(self) -> None`:
     - Open connection with WAL + busy_timeout.
     - Execute the same SQL as current `schema.py`:
       ```sql
       CREATE TABLE IF NOT EXISTS ml_cursor (
           id            INTEGER PRIMARY KEY CHECK (id = 1),
           last_event_id INTEGER NOT NULL DEFAULT 0,
           updated_at    INTEGER NOT NULL DEFAULT 0
       );
       INSERT OR IGNORE INTO ml_cursor (id, last_event_id, updated_at)
       VALUES (1, 0, 0);
       ```
     - Log success.
  2. This is a direct lift from `schema.py:ensure_ml_tables`.

- **Files**: `src/sigil_ml/store_sqlite.py` (add method)
- **Parallel?**: Yes -- can proceed once T001 is complete.
- **Notes**: `schema.py` is NOT removed in this WP (that happens in WP05). This subtask adds the equivalent method to `SqliteStore`.

### Subtask T007 -- Add create_store() factory function

- **Purpose**: Provide a single entry point that reads configuration and returns the appropriate `DataStore` implementation (FR-008).
- **Steps**:
  1. In `src/sigil_ml/store.py`, add:
     ```python
     def create_store() -> DataStore:
         """Create a DataStore instance based on the current configuration.

         Returns SqliteStore for local mode, PostgresStore for cloud mode.
         For now, only SqliteStore is implemented.
         """
         from sigil_ml import config
         from sigil_ml.store_sqlite import SqliteStore
         return SqliteStore(config.db_path())
     ```
  2. The factory will be extended in WP04 to support Postgres and in WP05 to read operating mode.
  3. Export `DataStore` and `create_store` from the module's `__all__`.

- **Files**: `src/sigil_ml/store.py` (add function, ~15 lines)
- **Parallel?**: No -- should be done after T001 and at least T002 exist.
- **Notes**: The factory uses lazy imports to avoid circular dependencies.

## Risks & Mitigations

- **Risk**: Protocol may need revision when WP02-WP03 discover additional query patterns. **Mitigation**: The SQL audit in this prompt is exhaustive (every `sqlite3` call in the codebase is listed). If any were missed, the protocol is easy to extend.
- **Risk**: Connection-per-method in `SqliteStore` may differ from the poller's current single-connection-per-poll-cycle pattern. **Mitigation**: WP02 will address transaction management. The SqliteStore methods work correctly in isolation; the poller adaptation handles batching.

## Review Guidance

- Verify the `DataStore` protocol covers every operation listed in the SQL audit table.
- Verify `SqliteStore._connect()` sets WAL mode and busy_timeout=5000.
- Verify `SqliteStore` methods return the exact same data shapes as current inline queries.
- Verify `create_store()` returns a valid `SqliteStore` instance.
- Confirm no `sqlite3` usage is introduced outside `store_sqlite.py`.

## Activity Log

- 2026-03-29T16:29:57Z -- system -- lane=planned -- Prompt created.
