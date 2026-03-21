"""Tests for feature extraction from SQLite."""

import json
import sqlite3
import time
from pathlib import Path

import pytest

from sigil_ml.features import (
    _query_events_for_task,
    _query_task,
    extract_duration_features,
    extract_stuck_features,
)


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    """Create a temporary SQLite DB with the sigild schema and sample data."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))

    conn.executescript("""
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            repo_root TEXT,
            branch TEXT,
            phase TEXT,
            files TEXT,
            started_at INTEGER,
            last_active INTEGER,
            completed_at INTEGER,
            commit_count INTEGER,
            test_runs INTEGER,
            test_fails INTEGER
        );

        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT,
            source TEXT,
            payload TEXT,
            ts INTEGER
        );
    """)

    now_ms = int(time.time() * 1000)

    # Insert a task
    conn.execute(
        "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "task-1",
            "/tmp/repo",
            "feature/add-widget",
            "coding",
            json.dumps({"main.py": 5, "utils.py": 3}),
            now_ms - 3600_000,  # started 1 hour ago
            now_ms - 60_000,  # last active 1 min ago
            None,  # not completed
            2,
            5,
            3,
        ),
    )

    # Insert a completed task
    conn.execute(
        "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "task-2",
            "/tmp/repo",
            "fix/bug",
            "testing",
            json.dumps({"fix.py": 2}),
            now_ms - 7200_000,
            now_ms - 300_000,
            now_ms - 300_000,
            3,
            10,
            1,
        ),
    )

    # Insert events for task-1
    events = [
        ("edit", "editor", json.dumps({"file": "main.py"}), now_ms - 3000_000),
        ("edit", "editor", json.dumps({"file": "main.py"}), now_ms - 2800_000),
        ("edit", "editor", json.dumps({"file": "utils.py"}), now_ms - 2600_000),
        ("save", "editor", json.dumps({"file": "main.py"}), now_ms - 2500_000),
        ("commit", "git", json.dumps({"hash": "abc123"}), now_ms - 1800_000),
        ("phase_change", "sigild", json.dumps({"phase": "coding"}), now_ms - 1200_000),
        ("edit", "editor", json.dumps({"file": "main.py"}), now_ms - 600_000),
    ]
    conn.executemany(
        "INSERT INTO events (kind, source, payload, ts) VALUES (?, ?, ?, ?)",
        events,
    )

    conn.commit()
    conn.close()
    return db_path


class TestQueryHelpers:
    def test_query_task_exists(self, test_db: Path) -> None:
        task = _query_task(test_db, "task-1")
        assert task is not None
        assert task["branch"] == "feature/add-widget"
        assert task["phase"] == "coding"

    def test_query_task_not_found(self, test_db: Path) -> None:
        assert _query_task(test_db, "nonexistent") is None

    def test_query_events(self, test_db: Path) -> None:
        events = _query_events_for_task(test_db, "task-1")
        assert len(events) > 0
        kinds = [e["kind"] for e in events]
        assert "edit" in kinds
        assert "commit" in kinds


class TestStuckFeatures:
    def test_basic_extraction(self, test_db: Path) -> None:
        features = extract_stuck_features(test_db, "task-1")
        assert "test_failure_count" in features
        assert "time_in_phase_sec" in features
        assert "edit_velocity" in features
        assert "file_switch_rate" in features
        assert "session_length_sec" in features
        assert "time_since_last_commit_sec" in features

        assert features["test_failure_count"] == 3.0
        assert features["session_length_sec"] > 0
        assert features["edit_velocity"] >= 0

    def test_missing_task(self, test_db: Path) -> None:
        features = extract_stuck_features(test_db, "nonexistent")
        assert features["test_failure_count"] == 0.0
        assert features["session_length_sec"] == 0.0

    def test_file_switch_rate(self, test_db: Path) -> None:
        features = extract_stuck_features(test_db, "task-1")
        # We have edits to main.py and utils.py
        assert 0.0 < features["file_switch_rate"] <= 1.0


class TestDurationFeatures:
    def test_basic_extraction(self, test_db: Path) -> None:
        features = extract_duration_features(test_db, "task-1")
        assert features["file_count"] == 2.0
        assert features["total_edits"] >= 0
        assert 0 <= features["time_of_day_hour"] < 24
        assert features["branch_name_length"] == len("feature/add-widget")

    def test_missing_task(self, test_db: Path) -> None:
        features = extract_duration_features(test_db, "nonexistent")
        assert features["file_count"] == 0.0
        assert features["branch_name_length"] == 0.0
