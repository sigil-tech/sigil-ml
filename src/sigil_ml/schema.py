"""Bootstrap Python-owned SQLite tables.

Go owns and migrates the main schema. Python only creates what it
exclusively manages: the polling cursor.

Call ensure_ml_tables() on server startup before the poller starts.
All statements are idempotent (IF NOT EXISTS).
"""
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_ml_tables(db_path: Path) -> None:
    """Create Python-owned tables if they don't exist."""
    conn = sqlite3.connect(str(db_path), timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS ml_cursor (
                id            INTEGER PRIMARY KEY CHECK (id = 1),
                last_event_id INTEGER NOT NULL DEFAULT 0,
                updated_at    INTEGER NOT NULL DEFAULT 0
            );
            INSERT OR IGNORE INTO ml_cursor (id, last_event_id, updated_at)
            VALUES (1, 0, 0);
        """)
        conn.commit()
        logger.info("schema: ml_cursor table ensured")
    finally:
        conn.close()
