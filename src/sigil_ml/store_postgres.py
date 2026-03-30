"""DataStore implementation backed by PostgreSQL.

Supports per-tenant schema isolation. Each tenant's tables live
in a dedicated Postgres schema (e.g., tenant_abc.events).

Requires: pip install sigil-ml[cloud]
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class PostgresStore:
    """DataStore implementation backed by a PostgreSQL database.

    Args:
        connection_url: Postgres connection URL (e.g., postgresql://user:pass@host:5432/dbname)
        tenant: Tenant identifier for schema isolation. Defaults to "public".
    """

    def __init__(self, connection_url: str, tenant: str = "public") -> None:
        try:
            import psycopg2
            from psycopg2 import sql as pg_sql
        except ImportError:
            raise ImportError(
                "psycopg2-binary is required for PostgresStore. "
                "Install with: pip install sigil-ml[cloud]"
            ) from None

        self._connection_url = connection_url
        self._tenant = tenant
        self._conn = None
        self._psycopg2 = psycopg2
        self._sql = pg_sql

    # --- Connection lifecycle ---

    def _get_conn(self):
        """Return existing connection or create a new one with tenant schema."""
        if self._conn is None or self._conn.closed:
            self._conn = self._psycopg2.connect(self._connection_url)
            self._conn.autocommit = False
            with self._conn.cursor() as cur:
                cur.execute(
                    self._sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                        self._sql.Identifier(self._tenant)
                    )
                )
                cur.execute(
                    self._sql.SQL("SET search_path TO {}, public").format(
                        self._sql.Identifier(self._tenant)
                    )
                )
            self._conn.commit()
        return self._conn

    def commit(self) -> None:
        """Commit the current transaction."""
        if self._conn and not self._conn.closed:
            self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None

    # --- Schema bootstrap ---

    def ensure_tables(self) -> None:
        """Create Python-owned tables in tenant schema if they don't exist."""
        conn = self._get_conn()
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
        logger.info("postgres: ml_cursor table ensured in schema %s", self._tenant)

    # --- Cursor operations ---

    def get_cursor(self) -> int:
        """Return the last processed event ID from ml_cursor."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT last_event_id FROM ml_cursor WHERE id = 1")
            row = cur.fetchone()
            return row[0] if row else 0

    def update_cursor(self, event_id: int) -> None:
        """Update ml_cursor.last_event_id to the given event_id."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE ml_cursor SET last_event_id = %s, updated_at = %s WHERE id = 1",
                (event_id, int(time.time() * 1000)),
            )

    # --- Event queries ---

    def get_events_since(self, since_id: int, limit: int = 100) -> list[dict[str, Any]]:
        """Return events with id > since_id, ordered by id ASC, up to limit."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, kind, source, payload, ts FROM events "
                "WHERE id > %s ORDER BY id ASC LIMIT %s",
                (since_id, limit),
            )
            columns = ["id", "kind", "source", "payload", "ts"]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def get_events_for_task(self, task_id: str, since: int | None = None) -> list[dict[str, Any]]:
        """Return events within a task's time window, with JSON payloads parsed."""
        task = self.get_task_by_id(task_id)
        if task is None:
            return []

        start = since if since is not None else task.get("started_at", 0)
        end = task.get("completed_at") or task.get("last_active") or int(time.time() * 1000)

        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM events WHERE ts >= %s AND ts <= %s ORDER BY ts",
                (start, end),
            )
            if cur.description is None:
                return []
            columns = [desc[0] for desc in cur.description]
            rows = [dict(zip(columns, row)) for row in cur.fetchall()]

        # Parse JSON payload
        for row in rows:
            if isinstance(row.get("payload"), str):
                try:
                    row["payload"] = json.loads(row["payload"])
                except (json.JSONDecodeError, TypeError):
                    pass
        return rows

    # --- Task queries ---

    def get_active_task(self) -> str | None:
        """Return the ID of the active (non-idle, not completed) task, or None."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM tasks WHERE phase != 'idle' "
                "AND completed_at IS NULL ORDER BY last_active DESC LIMIT 1"
            )
            row = cur.fetchone()
            return row[0] if row else None

    def get_task_by_id(self, task_id: str) -> dict[str, Any] | None:
        """Return a full task row as a dict, or None if not found."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM tasks WHERE id = %s", (task_id,))
            if cur.description is None:
                return None
            columns = [desc[0] for desc in cur.description]
            row = cur.fetchone()
            if row is None:
                return None
            return dict(zip(columns, row))

    def get_session_info(self, task_id: str) -> dict[str, Any] | None:
        """Return started_at, phase, test_fails for a task."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT started_at, phase, test_fails FROM tasks WHERE id = %s",
                (task_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return {"started_at": row[0], "phase": row[1], "test_fails": row[2]}

    def get_quality_task_stats(self) -> dict[str, Any] | None:
        """Return test_runs, test_fails, commit_count from the most recently completed task."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT test_runs, test_fails, commit_count FROM tasks "
                "WHERE completed_at IS NOT NULL ORDER BY completed_at DESC LIMIT 1"
            )
            row = cur.fetchone()
            if row is None:
                return None
            return {"test_runs": row[0], "test_fails": row[1], "commit_count": row[2]}

    def get_completed_task_ids(self) -> list[str]:
        """Return IDs of all completed tasks."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM tasks WHERE completed_at IS NOT NULL")
            return [row[0] for row in cur.fetchall()]

    def get_completed_tasks_with_timestamps(self) -> list[dict[str, Any]]:
        """Return id, started_at, completed_at for completed tasks with both timestamps set."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, started_at, completed_at FROM tasks "
                "WHERE completed_at IS NOT NULL AND started_at IS NOT NULL"
            )
            return [
                {"id": row[0], "started_at": row[1], "completed_at": row[2]}
                for row in cur.fetchall()
            ]

    def count_completed_tasks(self) -> int:
        """Return the count of completed tasks."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM tasks WHERE completed_at IS NOT NULL")
            row = cur.fetchone()
            return row[0] if row else 0

    # --- Status data ---

    def get_status_data(self) -> dict[str, Any]:
        """Return cursor info and latest non-expired predictions for /status."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT last_event_id, updated_at FROM ml_cursor WHERE id = 1")
            cursor_row = cur.fetchone()
            cursor_data = None
            if cursor_row:
                cursor_data = {"last_event_id": cursor_row[0], "updated_at": cursor_row[1]}

            now_ms = int(time.time() * 1000)
            cur.execute(
                "SELECT model, confidence, created_at FROM ml_predictions "
                "WHERE expires_at IS NULL OR expires_at > %s "
                "ORDER BY created_at DESC",
                (now_ms,),
            )
            preds = [
                {"model": row[0], "confidence": row[1], "created_at": row[2]}
                for row in cur.fetchall()
            ]

        return {
            "cursor": cursor_data,
            "latest_predictions": preds,
        }

    # --- Write operations (ml_predictions, ml_events only) ---

    def insert_prediction(
        self, model: str, result: dict, confidence: float, ttl_sec: int | None
    ) -> None:
        """Insert a row into ml_predictions."""
        conn = self._get_conn()
        now_ms = int(time.time() * 1000)
        expires_ms = (now_ms + ttl_sec * 1000) if ttl_sec else None
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ml_predictions (model, result, confidence, created_at, expires_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (model, json.dumps(result), round(confidence, 4), now_ms, expires_ms),
            )

    def insert_ml_event(
        self, kind: str, endpoint: str, routing: str, latency_ms: int
    ) -> None:
        """Insert a row into ml_events."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ml_events (kind, endpoint, routing, latency_ms, ts) "
                "VALUES (%s, %s, %s, %s, %s)",
                (kind, endpoint, routing, latency_ms, int(time.time() * 1000)),
            )
