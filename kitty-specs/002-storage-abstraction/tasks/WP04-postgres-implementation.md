---
work_package_id: WP04
title: Postgres Implementation
lane: planned
dependencies: [WP01]
subtasks:
- T022
- T023
- T024
- T025
- T026
- T027
- T028
phase: Phase 3 - Cloud Backend
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
- FR-004
- FR-008
- FR-009
- FR-011
- FR-012
---

# Work Package Prompt: WP04 -- Postgres Implementation

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

This WP depends on WP01 only (the DataStore protocol). It can be implemented in parallel with WP02 and WP03.

```bash
spec-kitty implement WP04 --base WP01
```

---

## Objectives & Success Criteria

- Implement `PostgresStore` that satisfies the `DataStore` protocol with a Postgres backend.
- Support per-tenant schema isolation for multi-tenant cloud deployment.
- Accept connection configuration via `SIGIL_POSTGRES_URL` environment variable.
- Respect table ownership rules: Python only writes to `ml_predictions`, `ml_events`, `ml_cursor`.
- **Success**: `PostgresStore` can be instantiated with a Postgres connection URL, and all DataStore methods work correctly against a Postgres database. Data formats are identical to `SqliteStore` output.

## Context & Constraints

- **Spec**: `kitty-specs/002-storage-abstraction/spec.md` -- FR-004, FR-008, FR-009, FR-011, FR-012
- **Prerequisite**: WP01 (DataStore protocol) is complete. `DataStore` protocol exists in `src/sigil_ml/store.py`.
- **Dependency constraint**: `sigil-ml` limits dependencies to `scikit-learn`, `numpy`, `fastapi`, `uvicorn`, `joblib`. Adding `psycopg2-binary` is acceptable because it is a standard lightweight driver. Do NOT add SQLAlchemy or other ORMs.
- **Table ownership** (from CLAUDE.md):
  - Python READS: `events`, `tasks`, `patterns`, `suggestions`
  - Python WRITES: `ml_predictions`, `ml_events`, `ml_cursor`
- **Schema mirroring**: The Postgres tables must have the same columns and types as the SQLite schema so that query results are in the same format. The Go daemon owns the main schema creation, but Python must create its owned tables (`ml_cursor`).
- **Per-tenant isolation**: In cloud mode, each tenant gets a Postgres schema (e.g., `SET search_path TO tenant_abc`). The tenant identifier comes from a configuration parameter.

## Subtasks & Detailed Guidance

### Subtask T022 -- Add psycopg2-binary to project dependencies

- **Purpose**: Add the Postgres driver as a dependency so `PostgresStore` can connect to Postgres.
- **Steps**:
  1. Check the project's dependency file. Look for `pyproject.toml` or `setup.py` or `requirements.txt`:
     ```bash
     ls -la pyproject.toml setup.py setup.cfg requirements*.txt
     ```
  2. Add `psycopg2-binary` as an **optional** dependency (not required for local-only mode):
     - In `pyproject.toml` (if using extras):
       ```toml
       [project.optional-dependencies]
       postgres = ["psycopg2-binary>=2.9"]
       ```
     - Or in `requirements.txt`, add with a comment:
       ```
       # Optional: only needed for cloud/Postgres mode
       psycopg2-binary>=2.9
       ```
  3. Do NOT make it a hard dependency. The SQLite backend must work without psycopg2 installed.
  4. Verify the import will be lazy (only when `PostgresStore` is instantiated, not at module level).

- **Files**: `pyproject.toml` or `requirements.txt` (modify)
- **Parallel?**: No -- do first so the driver is available for T023+.
- **Notes**: `psycopg2-binary` is preferred over `psycopg2` because it includes pre-built binaries and avoids requiring `libpq-dev` on the build machine.

### Subtask T023 -- Implement PostgresStore connection management and tenant schema isolation

- **Purpose**: Create the `PostgresStore` class with connection handling, tenant schema support, and the foundation for all data operations.
- **Steps**:
  1. Create `src/sigil_ml/store_postgres.py`.
  2. Import `psycopg2` lazily (inside the class or a try/except at module level):
     ```python
     try:
         import psycopg2
         import psycopg2.extras
     except ImportError:
         psycopg2 = None  # type: ignore
     ```
  3. Define the class:
     ```python
     class PostgresStore:
         """DataStore implementation backed by PostgreSQL.

         Supports per-tenant schema isolation via Postgres search_path.
         """

         def __init__(self, connection_url: str, tenant: str = "default") -> None:
             if psycopg2 is None:
                 raise ImportError(
                     "psycopg2-binary is required for Postgres support. "
                     "Install with: pip install sigil-ml[postgres]"
                 )
             self._connection_url = connection_url
             self._tenant = tenant
             self._schema = f"tenant_{tenant}"
     ```
  4. Implement `_connect(self)`:
     ```python
     def _connect(self):
         conn = psycopg2.connect(self._connection_url)
         conn.autocommit = False
         with conn.cursor() as cur:
             # Ensure tenant schema exists
             cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self._schema}")
             cur.execute(f"SET search_path TO {self._schema}, public")
             conn.commit()
         return conn
     ```
  5. **Security note**: The schema name is derived from the tenant identifier. Validate that the tenant string contains only alphanumeric characters and underscores to prevent SQL injection:
     ```python
     import re
     if not re.match(r'^[a-zA-Z0-9_]+$', tenant):
         raise ValueError(f"Invalid tenant identifier: {tenant}")
     ```
  6. Consider connection pooling for production use, but a simple connect-per-call approach is acceptable for initial implementation. Add a TODO comment for future pooling.

- **Files**: `src/sigil_ml/store_postgres.py` (new file, ~60 lines for connection management)
- **Parallel?**: No -- this must be done first as T024-T026 depend on `_connect`.
- **Notes**: Postgres does not have `PRAGMA` equivalents. WAL mode is not applicable. The `busy_timeout` equivalent is handled by `psycopg2`'s connection timeout parameter or `statement_timeout`.

### Subtask T024 -- Implement PostgresStore read operations

- **Purpose**: Implement all read-only DataStore methods against Postgres.
- **Steps**:
  1. Implement `get_cursor(self) -> int`:
     ```python
     def get_cursor(self) -> int:
         conn = self._connect()
         try:
             with conn.cursor() as cur:
                 cur.execute("SELECT last_event_id FROM ml_cursor WHERE id = 1")
                 row = cur.fetchone()
                 return row[0] if row else 0
         finally:
             conn.close()
     ```

  2. Implement `get_cursor_status(self) -> dict | None` (same pattern, returns dict).

  3. Implement `get_events_since(self, since_id: int, limit: int = 100) -> list[dict]`:
     ```python
     def get_events_since(self, since_id: int, limit: int = 100) -> list[dict]:
         conn = self._connect()
         try:
             with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                 cur.execute(
                     "SELECT id, kind, source, payload, ts FROM events "
                     "WHERE id > %s ORDER BY id ASC LIMIT %s",
                     (since_id, limit),
                 )
                 rows = cur.fetchall()
             # Parse JSON payload
             for row in rows:
                 if isinstance(row.get("payload"), str):
                     try:
                         row["payload"] = json.loads(row["payload"])
                     except (json.JSONDecodeError, TypeError):
                         pass
             return [dict(r) for r in rows]
         finally:
             conn.close()
     ```

     **Important**: Postgres may store `payload` as `jsonb` natively (not a string). Check the column type. If `jsonb`, the driver returns it as a Python dict already -- no parsing needed. Handle both cases.

  4. Implement `get_active_task(self) -> str | None`:
     ```python
     def get_active_task(self) -> str | None:
         conn = self._connect()
         try:
             with conn.cursor() as cur:
                 cur.execute(
                     "SELECT id FROM tasks WHERE phase != 'idle' AND completed_at IS NULL "
                     "ORDER BY last_active DESC LIMIT 1"
                 )
                 row = cur.fetchone()
                 return row[0] if row else None
         finally:
             conn.close()
     ```

  5. Implement `get_task_by_id`, `get_task_session_info`, `get_events_for_task`, `get_completed_task_ids`, `get_completed_tasks_with_timing`, `count_completed_tasks`, `get_quality_task_stats`, `get_active_predictions` -- all follow the same pattern: connect, query with `%s` placeholders (Postgres style, not `?`), return results, close.

  6. **SQL syntax differences from SQLite**:
     - Placeholders: `%s` instead of `?`
     - `SELECT *` with dict cursor returns dicts natively (use `RealDictCursor`)
     - `INTEGER PRIMARY KEY` does not auto-increment in Postgres; use `SERIAL` or `GENERATED ALWAYS AS IDENTITY`
     - Boolean values: Postgres uses `TRUE`/`FALSE`, SQLite uses `1`/`0` -- but the DataStore returns Python types so this is handled by the driver

- **Files**: `src/sigil_ml/store_postgres.py` (add ~150 lines of read methods)
- **Parallel?**: Yes -- can proceed once T023 establishes connection management.
- **Notes**: Use `psycopg2.extras.RealDictCursor` for methods that return dicts. Use a regular cursor for scalar queries.

### Subtask T025 -- Implement PostgresStore write operations

- **Purpose**: Implement prediction writing, audit event logging, and cursor updates against Postgres.
- **Steps**:
  1. Implement `update_cursor(self, last_event_id: int) -> None`:
     ```python
     def update_cursor(self, last_event_id: int) -> None:
         conn = self._connect()
         try:
             with conn.cursor() as cur:
                 cur.execute(
                     "UPDATE ml_cursor SET last_event_id = %s, updated_at = %s WHERE id = 1",
                     (last_event_id, int(time.time() * 1000)),
                 )
             conn.commit()
         finally:
             conn.close()
     ```

  2. Implement `insert_prediction(self, model, result, confidence, ttl_sec)`:
     ```python
     def insert_prediction(self, model: str, result: dict, confidence: float, ttl_sec: int | None) -> None:
         now_ms = int(time.time() * 1000)
         expires_ms = (now_ms + ttl_sec * 1000) if ttl_sec else None
         conn = self._connect()
         try:
             with conn.cursor() as cur:
                 cur.execute(
                     "INSERT INTO ml_predictions (model, result, confidence, created_at, expires_at) "
                     "VALUES (%s, %s, %s, %s, %s)",
                     (model, json.dumps(result), round(confidence, 4), now_ms, expires_ms),
                 )
             conn.commit()
         finally:
             conn.close()
     ```

  3. Implement `insert_ml_event(self, kind, endpoint, routing, latency_ms)`:
     ```python
     def insert_ml_event(self, kind: str, endpoint: str, routing: str, latency_ms: int) -> None:
         conn = self._connect()
         try:
             with conn.cursor() as cur:
                 cur.execute(
                     "INSERT INTO ml_events (kind, endpoint, routing, latency_ms, ts) "
                     "VALUES (%s, %s, %s, %s, %s)",
                     (kind, endpoint, routing, latency_ms, int(time.time() * 1000)),
                 )
             conn.commit()
         finally:
             conn.close()
     ```

  4. **Table ownership enforcement**: The `PostgresStore` must only write to `ml_predictions`, `ml_events`, and `ml_cursor`. Verify that no write method targets `events`, `tasks`, `patterns`, or `suggestions` (FR-012).

- **Files**: `src/sigil_ml/store_postgres.py` (add ~80 lines of write methods)
- **Parallel?**: Yes -- can proceed once T023 establishes connection management.
- **Notes**: Postgres requires explicit `conn.commit()` after writes (autocommit is off). The SQLite store also commits explicitly, so the pattern is consistent.

### Subtask T026 -- Implement PostgresStore schema bootstrap

- **Purpose**: Create Python-owned tables in the tenant's Postgres schema on first access.
- **Steps**:
  1. Implement `ensure_tables(self) -> None`:
     ```python
     def ensure_tables(self) -> None:
         conn = self._connect()
         try:
             with conn.cursor() as cur:
                 cur.execute("""
                     CREATE TABLE IF NOT EXISTS ml_cursor (
                         id            INTEGER PRIMARY KEY CHECK (id = 1),
                         last_event_id BIGINT NOT NULL DEFAULT 0,
                         updated_at    BIGINT NOT NULL DEFAULT 0
                     )
                 """)
                 cur.execute("""
                     INSERT INTO ml_cursor (id, last_event_id, updated_at)
                     VALUES (1, 0, 0)
                     ON CONFLICT (id) DO NOTHING
                 """)
             conn.commit()
             logger.info("schema: ml_cursor table ensured in schema %s", self._schema)
         finally:
             conn.close()
     ```

  2. **SQL differences from SQLite**:
     - `INSERT OR IGNORE` (SQLite) becomes `INSERT ... ON CONFLICT (id) DO NOTHING` (Postgres)
     - `INTEGER NOT NULL DEFAULT 0` works in both
     - No `PRAGMA` statements needed

  3. The tenant schema itself is created in `_connect()` (T023). This method only creates tables within that schema.

  4. **Note**: Python does NOT create `ml_predictions` or `ml_events` tables. The Go daemon owns those schemas. Python only creates `ml_cursor` (same as SQLite behavior where only `ml_cursor` is created by `schema.py`).

- **Files**: `src/sigil_ml/store_postgres.py` (add ~30 lines)
- **Parallel?**: No -- should be done after T023.
- **Notes**: If the Go daemon hasn't created `ml_predictions` or `ml_events` yet, write operations will fail with a clear Postgres error. This is expected -- Python depends on Go having initialized the main schema.

### Subtask T027 -- Update create_store() factory for Postgres

- **Purpose**: Extend the factory function to return a `PostgresStore` when the operating mode is cloud.
- **Steps**:
  1. Open `src/sigil_ml/store.py`.
  2. Update `create_store()`:
     ```python
     def create_store() -> DataStore:
         """Create a DataStore based on the current operating mode.

         Local mode: SqliteStore backed by the sigild database file.
         Cloud mode: PostgresStore backed by a Postgres connection URL.
         """
         from sigil_ml import config

         mode = os.environ.get("SIGIL_MODE", "local")

         if mode == "cloud":
             postgres_url = os.environ.get("SIGIL_POSTGRES_URL")
             if not postgres_url:
                 raise ValueError("SIGIL_POSTGRES_URL is required in cloud mode")
             tenant = os.environ.get("SIGIL_TENANT", "default")
             from sigil_ml.store_postgres import PostgresStore
             return PostgresStore(postgres_url, tenant=tenant)
         else:
             from sigil_ml.store_sqlite import SqliteStore
             return SqliteStore(config.db_path())
     ```
  3. Lazy imports ensure that `psycopg2` is only imported when cloud mode is selected.

- **Files**: `src/sigil_ml/store.py` (modify `create_store`)
- **Parallel?**: No -- depends on T023-T026 being complete.
- **Notes**: The `SIGIL_MODE` environment variable may be introduced by feature 001 (cloud serving mode). If that feature isn't merged yet, this still works -- the env var defaults to "local".

### Subtask T028 -- Add Postgres connection configuration

- **Purpose**: Add environment variable support for Postgres connection URL and tenant identifier to the config module.
- **Steps**:
  1. Open `src/sigil_ml/config.py`.
  2. Add configuration functions:
     ```python
     def operating_mode() -> str:
         """Return the operating mode: 'local' or 'cloud'."""
         return os.environ.get("SIGIL_MODE", "local")

     def postgres_url() -> str | None:
         """Return the Postgres connection URL, or None if not configured."""
         return os.environ.get("SIGIL_POSTGRES_URL")

     def tenant_id() -> str:
         """Return the tenant identifier for multi-tenant Postgres deployment."""
         return os.environ.get("SIGIL_TENANT", "default")
     ```
  3. Update `create_store()` in `store.py` to use these config functions instead of reading `os.environ` directly:
     ```python
     from sigil_ml import config
     mode = config.operating_mode()
     if mode == "cloud":
         url = config.postgres_url()
         if not url:
             raise ValueError("SIGIL_POSTGRES_URL is required in cloud mode")
         from sigil_ml.store_postgres import PostgresStore
         return PostgresStore(url, tenant=config.tenant_id())
     ```

- **Files**: `src/sigil_ml/config.py` (add functions), `src/sigil_ml/store.py` (update `create_store`)
- **Parallel?**: Yes -- the config additions can be done alongside T023-T026.
- **Notes**: FR-009 requires the Postgres URL be configurable via environment variable. This is the standard pattern for 12-factor apps.

## Risks & Mitigations

- **Risk**: Postgres SQL dialect differences cause subtle bugs (e.g., `LIMIT` behavior, `NULL` handling, timestamp formats). **Mitigation**: The queries are simple CRUD. The prompt specifies Postgres-specific syntax (`%s` placeholders, `ON CONFLICT`, `RealDictCursor`).
- **Risk**: `psycopg2-binary` not available in all deployment environments. **Mitigation**: Made optional -- only imported when cloud mode is requested. Clear error message if missing.
- **Risk**: Tenant schema creation fails due to insufficient Postgres permissions. **Mitigation**: The error from `CREATE SCHEMA IF NOT EXISTS` will be clear. Document required permissions: the Postgres user needs `CREATE` on the database.
- **Risk**: The Go daemon may use different column types in Postgres vs SQLite (e.g., `jsonb` for payload). **Mitigation**: Handle both string and dict payload formats in read methods.

## Review Guidance

- Verify `PostgresStore` implements every method in the `DataStore` protocol.
- Verify no raw SQL uses `?` placeholders (must be `%s` for Postgres).
- Verify tenant schema name is validated (alphanumeric + underscore only).
- Verify `psycopg2` import is lazy (not at module level).
- Verify table ownership: no writes to `events`, `tasks`, `patterns`, `suggestions`.
- Verify `create_store()` returns `PostgresStore` when `SIGIL_MODE=cloud`.
- Test locally: `SIGIL_MODE=local python -c "from sigil_ml.store import create_store; s = create_store()"` should work without psycopg2 installed.

## Activity Log

- 2026-03-29T16:29:57Z -- system -- lane=planned -- Prompt created.
