"""Tests for SigilFeatureStore — Feast integration layer.

These tests cover:
- SigilFeatureStore initialization and apply()
- materialize_task() + retrieval roundtrip (get_stuck_features, get_duration_features)
- has_features() presence check
- Graceful fallback when Feast is not available
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from sigil_ml.store_sqlite import SqliteStore

# ---------------------------------------------------------------------------
# Availability guard — skip if feast is not installed
# ---------------------------------------------------------------------------


def _feast_importable() -> bool:
    try:
        import feast  # noqa: F401

        return True
    except ImportError:
        return False


feast_available = pytest.mark.skipif(
    not _feast_importable(),
    reason="feast not installed — run: pip install 'feast[sqlite]>=0.40'",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_db(tmp_path: Path) -> Path:
    """Minimal SQLite DB with task and events tables for feature extraction."""
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

    conn.execute(
        "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "task-feast-1",
            "/tmp/repo",
            "feat/feast-test",
            "coding",
            json.dumps({"main.py": 3, "utils.py": 1}),
            now_ms - 3600_000,
            now_ms - 60_000,
            None,
            1,
            4,
            2,
        ),
    )

    events = [
        ("edit", "editor", json.dumps({"file": "main.py"}), now_ms - 3000_000),
        ("edit", "editor", json.dumps({"file": "utils.py"}), now_ms - 2800_000),
        ("save", "editor", json.dumps({"file": "main.py"}), now_ms - 2500_000),
        ("commit", "git", json.dumps({"hash": "deadbeef"}), now_ms - 1800_000),
    ]
    conn.executemany(
        "INSERT INTO events (kind, source, payload, ts) VALUES (?, ?, ?, ?)",
        events,
    )

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture()
def sqlite_store(test_db: Path) -> SqliteStore:
    return SqliteStore(test_db)


@pytest.fixture()
def feast_repo_path(tmp_path: Path) -> str:
    """Return a temp feast repo path with feature_store.yaml configured to use
    tmp_path for both the registry and online store databases."""
    import importlib.resources
    import shutil

    # Copy the packaged feast_repo into a temp directory so each test gets
    # an isolated, writeable registry and online store.
    src = str(importlib.resources.files("sigil_ml") / "feast_repo")
    dst = str(tmp_path / "feast_repo")
    shutil.copytree(src, dst)

    # Overwrite feature_store.yaml to use tmp_path for data files
    data_dir = tmp_path / "feast_data"
    data_dir.mkdir()
    yaml_content = f"""\
project: sigil_ml
provider: local
registry:
  registry_type: file
  path: {data_dir}/registry.db
online_store:
  type: sqlite
  path: {data_dir}/online_store.db
entity_key_serialization_version: 3
"""
    (Path(dst) / "feature_store.yaml").write_text(yaml_content)
    return dst


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSigilFeatureStoreInit:
    @feast_available
    def test_init_with_explicit_repo_path(self, feast_repo_path: str) -> None:
        """SigilFeatureStore can be constructed with an explicit repo path."""
        from sigil_ml.feature_store import SigilFeatureStore

        fs = SigilFeatureStore(repo_path=feast_repo_path)
        assert fs._store is not None

    @feast_available
    def test_apply_registers_feature_views(self, feast_repo_path: str) -> None:
        """apply() registers entities and feature views with the Feast registry."""
        from sigil_ml.feature_store import SigilFeatureStore

        fs = SigilFeatureStore(repo_path=feast_repo_path)
        # apply() should not raise
        fs.apply()

        # Verify feature views are registered
        fvs = {fv.name for fv in fs._store.list_feature_views()}
        assert "stuck_features" in fvs
        assert "duration_features" in fvs
        assert "fleet_focus_features" in fvs


class TestMaterializeAndRetrieve:
    @feast_available
    def test_materialize_task_pushes_stuck_features(
        self, feast_repo_path: str, sqlite_store: SqliteStore
    ) -> None:
        """materialize_task() pushes stuck features into the online store."""
        from sigil_ml.feature_store import SigilFeatureStore

        fs = SigilFeatureStore(repo_path=feast_repo_path)
        fs.apply()

        fs.materialize_task(sqlite_store, "task-feast-1")

        stuck = fs.get_stuck_features("task-feast-1")
        assert "test_failure_count" in stuck
        assert "time_in_phase_sec" in stuck
        assert "edit_velocity" in stuck
        assert "file_switch_rate" in stuck
        assert "session_length_sec" in stuck
        assert "time_since_last_commit_sec" in stuck

    @feast_available
    def test_materialize_task_pushes_duration_features(
        self, feast_repo_path: str, sqlite_store: SqliteStore
    ) -> None:
        """materialize_task() pushes duration features into the online store."""
        from sigil_ml.feature_store import SigilFeatureStore

        fs = SigilFeatureStore(repo_path=feast_repo_path)
        fs.apply()

        fs.materialize_task(sqlite_store, "task-feast-1")

        duration = fs.get_duration_features("task-feast-1")
        assert "file_count" in duration
        assert "total_edits" in duration
        assert "time_of_day_hour" in duration
        assert "branch_name_length" in duration

    @feast_available
    def test_stuck_feature_values_match_extractor(
        self, feast_repo_path: str, sqlite_store: SqliteStore
    ) -> None:
        """Features retrieved from Feast match what the extractor computes directly."""
        from sigil_ml.feature_store import SigilFeatureStore
        from sigil_ml.features import extract_stuck_features

        fs = SigilFeatureStore(repo_path=feast_repo_path)
        fs.apply()

        direct = extract_stuck_features(sqlite_store, "task-feast-1")
        fs.materialize_task(sqlite_store, "task-feast-1")
        from_feast = fs.get_stuck_features("task-feast-1")

        # Values should match (within floating point tolerance for float64)
        for key in direct:
            assert key in from_feast, f"Missing key {key} in Feast response"
            assert from_feast[key] == pytest.approx(direct[key], rel=1e-6)

    @feast_available
    def test_duration_feature_values_match_extractor(
        self, feast_repo_path: str, sqlite_store: SqliteStore
    ) -> None:
        """Duration features retrieved from Feast match the extractor output."""
        from sigil_ml.feature_store import SigilFeatureStore
        from sigil_ml.features import extract_duration_features

        fs = SigilFeatureStore(repo_path=feast_repo_path)
        fs.apply()

        direct = extract_duration_features(sqlite_store, "task-feast-1")
        fs.materialize_task(sqlite_store, "task-feast-1")
        from_feast = fs.get_duration_features("task-feast-1")

        for key in direct:
            assert key in from_feast, f"Missing key {key} in Feast response"
            assert from_feast[key] == pytest.approx(direct[key], rel=1e-6)


class TestHasFeatures:
    @feast_available
    def test_has_features_false_before_materialization(self, feast_repo_path: str) -> None:
        """has_features() returns False for an entity that was never pushed."""
        from sigil_ml.feature_store import SigilFeatureStore

        fs = SigilFeatureStore(repo_path=feast_repo_path)
        fs.apply()

        assert fs.has_features("stuck_features", "nonexistent-task") is False

    @feast_available
    def test_has_features_true_after_materialization(
        self, feast_repo_path: str, sqlite_store: SqliteStore
    ) -> None:
        """has_features() returns True once the entity has been materialized."""
        from sigil_ml.feature_store import SigilFeatureStore

        fs = SigilFeatureStore(repo_path=feast_repo_path)
        fs.apply()

        fs.materialize_task(sqlite_store, "task-feast-1")
        assert fs.has_features("stuck_features", "task-feast-1") is True


class TestFallbackWhenFeastUnavailable:
    def test_import_error_is_handled_gracefully(self) -> None:
        """When feast cannot be imported, SigilFeatureStore raises ImportError — callers
        should catch it and leave feature_store=None (graceful degradation)."""
        with patch.dict("sys.modules", {"feast": None}), pytest.raises((ImportError, TypeError)):
            from sigil_ml.feature_store import SigilFeatureStore  # noqa: F401

            SigilFeatureStore()

    def test_app_state_feature_store_defaults_to_none(self) -> None:
        """AppState.feature_store starts as None — safe to check before calling."""
        from sigil_ml.app import AppState

        state = AppState()
        assert state.feature_store is None

    def test_routes_handle_none_feature_store(self, tmp_path) -> None:
        """Routes fall back gracefully when feature_store is None.

        When task_id is provided but feature_store is None AND store is None,
        the route returns the fallback response (not a 500 error).
        """
        import os

        os.environ["XDG_DATA_HOME"] = str(tmp_path)

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from sigil_ml.app import AppState
        from sigil_ml.config import ServingMode
        from sigil_ml.routes import register_routes

        # Build a minimal app with feature_store=None and store=None
        state = AppState(mode=ServingMode.LOCAL)
        state.feature_store = None
        state.store = None
        state.stuck = None  # No model loaded

        app = FastAPI()
        register_routes(app, state)

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/predict/stuck", json={"task_id": "some-task"})
        assert resp.status_code == 200
        data = resp.json()
        # Should return fallback values, not crash
        assert "probability" in data
        assert "confidence" in data
