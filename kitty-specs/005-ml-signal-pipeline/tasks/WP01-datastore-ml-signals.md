---
work_package_id: "WP01"
title: "DataStore Extension & ml_signals Table"
lane: "planned"
dependencies: []
subtasks:
  - "T001"
  - "T002"
  - "T003"
  - "T004"
  - "T005"
  - "T006"
phase: "Phase 0 - Foundation"
assignee: ""
agent: ""
shell_pid: ""
review_status: ""
reviewed_by: ""
requirement_refs:
  - "FR-007"
  - "FR-008"
  - "FR-017"
history:
  - timestamp: "2026-03-30T18:27:35Z"
    lane: "planned"
    agent: "system"
    shell_pid: ""
    action: "Prompt generated via /spec-kitty.tasks"
---

# Work Package Prompt: WP01 -- DataStore Extension & ml_signals Table

## IMPORTANT: Review Feedback Status

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

```bash
spec-kitty implement WP01
```

No dependencies (starting package).

---

## Objectives & Success Criteria

1. The `DataStore` protocol in `src/sigil_ml/store.py` gains two new methods: `insert_signal()` and `get_signal_feedback()`.
2. Both `SqliteStore` and `PostgresStore` implement these methods.
3. `ensure_tables()` in both stores creates the `ml_signals` table with appropriate indexes.
4. A `Signal` dataclass in `src/sigil_ml/signals/__init__.py` provides the shared contract for all signal models.
5. All existing DataStore behavior remains unchanged -- zero regression.
6. The `ml_signals` table follows the Python-owned convention (Python creates and writes; Go reads only).

## Context & Constraints

- **Spec**: FR-007 (structured signals to ml_signals), FR-008 (evidence, confidence, action, timestamps, expiry), FR-017 (Python-owned table).
- **Data Model**: See `kitty-specs/005-ml-signal-pipeline/data-model.md` for MLSignal entity definition and Evidence JSON structure.
- **Existing Patterns**: Follow `insert_prediction()` and `insert_ml_event()` patterns in both stores. The `ensure_tables()` pattern in `store_sqlite.py` uses `executescript()` with `CREATE TABLE IF NOT EXISTS`.
- **Protocol Extension**: The `DataStore` protocol in `store.py` is `@runtime_checkable`. New methods must be added with `...` (Ellipsis) bodies matching existing convention.
- **Signal ID Return**: `insert_signal()` must return the auto-generated integer ID so the SignalEngine can track it for rate limiting and logging.

---

## Subtasks & Detailed Guidance

### Subtask T001 -- Add `insert_signal()` to DataStore Protocol

- **Purpose**: Extend the DataStore protocol with a method to write structured signals to the `ml_signals` table.
- **Steps**:
  1. Open `src/sigil_ml/store.py`.
  2. Add the following method signature to the `DataStore` protocol class, after `insert_ml_event()`:
     ```python
     def insert_signal(
         self,
         signal_type: str,
         confidence: float,
         evidence: dict,
         suggested_action: str | None = None,
         ttl_sec: int | None = None,
     ) -> int:
         """Insert a signal into ml_signals. Returns the signal ID.

         Args:
             signal_type: Model-generated type (e.g., "velocity_deviation", "test_cadence_drop").
             confidence: Model's confidence score (0.0 to 1.0).
             evidence: Structured JSON evidence for LLM rendering.
             suggested_action: Optional generic action hint (e.g., "test", "commit").
             ttl_sec: Optional time-to-live in seconds. None = no expiry.

         Returns:
             The auto-generated integer ID of the inserted signal.
         """
         ...
     ```
  3. Update the module docstring comment about Python ownership to include `ml_signals`:
     ```
     Python only writes to ml_predictions, ml_events, ml_cursor, ml_signals.
     ```
- **Files**: `src/sigil_ml/store.py`
- **Parallel?**: No -- other subtasks depend on this signature.
- **Validation**:
  - [ ] Protocol type-checks correctly (`isinstance(store, DataStore)` returns True)
  - [ ] Signature matches plan.md D8 specification

### Subtask T002 -- Add `get_signal_feedback()` to DataStore Protocol

- **Purpose**: Extend the DataStore protocol with a method to read feedback linkages from the suggestions table for training labels.
- **Steps**:
  1. Add to the `DataStore` protocol, after `insert_signal()`:
     ```python
     def get_signal_feedback(self, since_ms: int) -> list[dict]:
         """Read feedback linkages from suggestions table for training.

         Returns rows where a suggestion was linked to an ml_signal
         (via signal_id column) and has a status of accepted/dismissed/ignored.

         Args:
             since_ms: Only return feedback newer than this Unix ms timestamp.

         Returns:
             List of dicts with keys: signal_id, signal_type, status, created_at.
         """
         ...
     ```
  2. This method reads from Go-owned `suggestions` table (SELECT only, following ownership rules).
- **Files**: `src/sigil_ml/store.py`
- **Parallel?**: Independent from T001 but both modify the same file.
- **Notes**: The Go daemon will add a `signal_id` column to the `suggestions` table (Feature 021 in sigil repo). For now, implement with a graceful fallback -- if the `signal_id` column doesn't exist, return an empty list.
- **Validation**:
  - [ ] Method signature is in the protocol
  - [ ] Returns empty list when suggestions table lacks signal_id column

### Subtask T003 -- Implement `insert_signal()` in SqliteStore

- **Purpose**: Implement the SQLite version of signal insertion.
- **Steps**:
  1. Open `src/sigil_ml/store_sqlite.py`.
  2. Add the implementation after `insert_ml_event()`:
     ```python
     def insert_signal(
         self,
         signal_type: str,
         confidence: float,
         evidence: dict,
         suggested_action: str | None = None,
         ttl_sec: int | None = None,
     ) -> int:
         """Insert a signal into ml_signals. Returns the signal ID."""
         conn = self._get_conn()
         now_ms = int(time.time() * 1000)
         expires_ms = (now_ms + ttl_sec * 1000) if ttl_sec else None
         cur = conn.execute(
             "INSERT INTO ml_signals "
             "(signal_type, confidence, evidence, suggested_action, created_at, expires_at) "
             "VALUES (?, ?, ?, ?, ?, ?)",
             (signal_type, round(confidence, 4), json.dumps(evidence),
              suggested_action, now_ms, expires_ms),
         )
         return cur.lastrowid
     ```
  3. Also implement `get_signal_feedback()`:
     ```python
     def get_signal_feedback(self, since_ms: int) -> list[dict]:
         """Read feedback linkages from suggestions table."""
         conn = self._get_conn()
         try:
             rows = conn.execute(
                 "SELECT s.signal_id, ms.signal_type, s.status, s.created_at "
                 "FROM suggestions s "
                 "JOIN ml_signals ms ON s.signal_id = ms.id "
                 "WHERE s.signal_id IS NOT NULL AND s.created_at > ? "
                 "ORDER BY s.created_at ASC",
                 (since_ms,),
             ).fetchall()
             return [
                 {"signal_id": r[0], "signal_type": r[1], "status": r[2], "created_at": r[3]}
                 for r in rows
             ]
         except Exception:
             # signal_id column may not exist yet (Go Feature 021)
             logger.debug("get_signal_feedback: suggestions.signal_id not available yet")
             return []
     ```
- **Files**: `src/sigil_ml/store_sqlite.py`
- **Parallel?**: Can proceed alongside T004 (different file).
- **Validation**:
  - [ ] `insert_signal()` returns an integer > 0
  - [ ] Row appears in `ml_signals` table with correct column values
  - [ ] `get_signal_feedback()` returns empty list when signal_id column is missing
  - [ ] JSON evidence is correctly serialized

### Subtask T004 -- Implement `insert_signal()` in PostgresStore

- **Purpose**: Implement the Postgres version of signal insertion, following the existing Postgres patterns (parameterized queries with `%s`, cursor context managers).
- **Steps**:
  1. Open `src/sigil_ml/store_postgres.py`.
  2. Add the implementation following the same pattern as `insert_prediction()`:
     ```python
     def insert_signal(
         self,
         signal_type: str,
         confidence: float,
         evidence: dict,
         suggested_action: str | None = None,
         ttl_sec: int | None = None,
     ) -> int:
         """Insert a signal into ml_signals. Returns the signal ID."""
         conn = self._get_conn()
         now_ms = int(time.time() * 1000)
         expires_ms = (now_ms + ttl_sec * 1000) if ttl_sec else None
         with conn.cursor() as cur:
             cur.execute(
                 "INSERT INTO ml_signals "
                 "(signal_type, confidence, evidence, suggested_action, created_at, expires_at) "
                 "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                 (signal_type, round(confidence, 4), json.dumps(evidence),
                  suggested_action, now_ms, expires_ms),
             )
             row = cur.fetchone()
             return row[0]
     ```
  3. Also implement `get_signal_feedback()` with the same graceful fallback pattern.
- **Files**: `src/sigil_ml/store_postgres.py`
- **Parallel?**: Can proceed alongside T003 (different file).
- **Notes**: Postgres uses `RETURNING id` instead of `lastrowid` to get the auto-generated ID.
- **Validation**:
  - [ ] Uses `%s` parameter style (not `?`)
  - [ ] Uses `RETURNING id` to get the signal ID
  - [ ] Follows cursor context manager pattern from existing Postgres methods

### Subtask T005 -- Update `ensure_tables()` for ml_signals Table

- **Purpose**: Create the `ml_signals` table schema in both SqliteStore and PostgresStore.
- **Steps**:
  1. In `src/sigil_ml/store_sqlite.py`, extend the `ensure_tables()` `executescript()`:
     ```python
     def ensure_tables(self) -> None:
         conn = self._get_conn()
         conn.executescript("""
             CREATE TABLE IF NOT EXISTS ml_cursor (
                 id            INTEGER PRIMARY KEY CHECK (id = 1),
                 last_event_id INTEGER NOT NULL DEFAULT 0,
                 updated_at    INTEGER NOT NULL DEFAULT 0
             );
             INSERT OR IGNORE INTO ml_cursor (id, last_event_id, updated_at)
             VALUES (1, 0, 0);

             CREATE TABLE IF NOT EXISTS ml_signals (
                 id              INTEGER PRIMARY KEY AUTOINCREMENT,
                 signal_type     TEXT    NOT NULL,
                 confidence      REAL    NOT NULL,
                 evidence        TEXT    NOT NULL,
                 suggested_action TEXT,
                 created_at      INTEGER NOT NULL,
                 expires_at      INTEGER,
                 rendered        INTEGER NOT NULL DEFAULT 0,
                 suggestion_id   INTEGER
             );
             CREATE INDEX IF NOT EXISTS idx_ml_signals_created_at ON ml_signals(created_at);
             CREATE INDEX IF NOT EXISTS idx_ml_signals_rendered ON ml_signals(rendered);
         """)
         conn.commit()
         logger.info("schema: ml_cursor and ml_signals tables ensured")
     ```
  2. In `src/sigil_ml/store_postgres.py`, add to `ensure_tables()`:
     ```python
     cur.execute("""
         CREATE TABLE IF NOT EXISTS ml_signals (
             id              SERIAL PRIMARY KEY,
             signal_type     TEXT    NOT NULL,
             confidence      REAL   NOT NULL,
             evidence        TEXT    NOT NULL,
             suggested_action TEXT,
             created_at      BIGINT NOT NULL,
             expires_at      BIGINT,
             rendered        INTEGER NOT NULL DEFAULT 0,
             suggestion_id   INTEGER
         )
     """)
     cur.execute("""
         CREATE INDEX IF NOT EXISTS idx_ml_signals_created_at ON ml_signals(created_at)
     """)
     cur.execute("""
         CREATE INDEX IF NOT EXISTS idx_ml_signals_rendered ON ml_signals(rendered)
     """)
     ```
- **Files**: `src/sigil_ml/store_sqlite.py`, `src/sigil_ml/store_postgres.py`
- **Parallel?**: No -- depends on schema design being finalized.
- **Notes**: Schema matches data-model.md MLSignal entity. `rendered` defaults to 0 (false). `suggestion_id` is nullable (populated by Go after rendering).
- **Validation**:
  - [ ] Table created on first call to `ensure_tables()`
  - [ ] Indexes created on `created_at` and `rendered`
  - [ ] Idempotent: calling `ensure_tables()` twice does not error
  - [ ] Existing `ml_cursor` table creation is preserved unchanged

### Subtask T006 -- Create Signal Dataclass

- **Purpose**: Define the `Signal` dataclass that all signal models produce and the SignalEngine consumes.
- **Steps**:
  1. Create `src/sigil_ml/signals/__init__.py`:
     ```python
     """ML Signal Pipeline -- event-driven behavioral signal detection.

     This package contains:
       - Signal: shared dataclass for all signal models
       - SignalEngine: orchestrator that runs all models per poll cycle
       - BehaviorProfile: incremental per-user behavioral profile
       - PatternDetector: z-score and Isolation Forest anomaly detection
       - NextActionPredictor: n-gram action sequence prediction
       - FileRecommender: co-occurrence file recommendation
     """

     from __future__ import annotations

     import time
     from dataclasses import dataclass, field
     from typing import Any


     @dataclass
     class Signal:
         """A structured signal emitted when a model detects something noteworthy.

         Produced by signal models (PatternDetector, NextActionPredictor, FileRecommender).
         Consumed by SignalEngine, which writes it to ml_signals via DataStore.
         """

         signal_type: str
         """Model-generated type, e.g. 'velocity_deviation', 'divergence_test', 'file_cluster'."""

         confidence: float
         """Model's confidence (0.0 to 1.0) that this signal is worth surfacing."""

         evidence: dict[str, Any]
         """Structured evidence for LLM rendering. Must include 'source_model' key."""

         suggested_action: str | None = None
         """Generic action hint for LLM: 'investigate', 'test', 'commit', 'take_break'."""

         ttl_sec: int | None = None
         """Time-to-live in seconds. None = no expiry."""

         created_at: int = field(default_factory=lambda: int(time.time() * 1000))
         """Unix ms timestamp. Auto-populated on creation."""
     ```
  2. Ensure the `signals/` directory is a proper Python package (the `__init__.py` serves this role).
  3. The Signal dataclass is deliberately simple -- it carries data from model to store. The `signal_type` is NOT a fixed enum; it is model-generated (FR-004).
- **Files**: `src/sigil_ml/signals/__init__.py` (new file)
- **Parallel?**: Yes -- can proceed independently of T001-T005.
- **Notes**: The `evidence` dict must include a `source_model` key per data-model.md. This is enforced by convention (documented), not by runtime validation.
- **Validation**:
  - [ ] `Signal(signal_type="test", confidence=0.8, evidence={"source_model": "test"})` creates successfully
  - [ ] `created_at` is auto-populated with current Unix ms timestamp
  - [ ] Can import: `from sigil_ml.signals import Signal`

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Go daemon not yet reading ml_signals | Expected | Low | Table exists and is populated; Go reads when Feature 021 lands |
| suggestions.signal_id column missing | Expected | Low | Graceful fallback to empty list in get_signal_feedback() |
| Schema migration on existing databases | Medium | Medium | CREATE TABLE IF NOT EXISTS is idempotent |
| Protocol extension breaks runtime_checkable | Low | High | Test with isinstance() check after adding methods |

## Review Guidance

- **Protocol consistency**: Verify `insert_signal()` signature matches plan.md D8 exactly. Compare parameter names and types.
- **Store parity**: Both SqliteStore and PostgresStore must implement identical behavior. Compare side-by-side.
- **Schema correctness**: Verify ml_signals schema matches data-model.md MLSignal entity. Column types, nullability, defaults.
- **No regression**: Verify `ensure_tables()` still creates `ml_cursor` correctly. Verify `insert_prediction()` and `insert_ml_event()` are unchanged.
- **Signal dataclass**: Verify it has no external dependencies (pure dataclass, stdlib only).

---

## Activity Log

- 2026-03-30T18:27:35Z -- system -- lane=planned -- Prompt generated via /spec-kitty.tasks.
