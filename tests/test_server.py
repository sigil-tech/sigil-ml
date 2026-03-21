"""Tests for the FastAPI server endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _isolate_models(tmp_path, monkeypatch):
    """Redirect model weights to a temp directory."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))


@pytest.fixture
def client():
    """Create a test client for the FastAPI app.

    Uses raise_server_exceptions=False to avoid leaking startup errors.
    The TestClient context manager triggers startup/shutdown events.
    """
    from sigil_ml.app import create_app

    application = create_app()
    with TestClient(application) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "models" in data
        assert "uptime_sec" in data

    def test_health_reports_model_status(self, client: TestClient) -> None:
        resp = client.get("/health")
        data = resp.json()
        models = data["models"]
        assert "stuck" in models
        assert "activity" in models
        assert "workflow" in models
        assert "duration" in models
        # Models should be untrained since we use a clean temp dir
        assert models["stuck"] == "untrained"
        assert models["duration"] == "untrained"


class TestStuckEndpoint:
    def test_predict_with_features(self, client: TestClient) -> None:
        resp = client.post(
            "/predict/stuck",
            json={
                "features": {
                    "test_failure_count": 5,
                    "time_in_phase_sec": 1200,
                    "edit_velocity": 4.0,
                    "file_switch_rate": 0.7,
                    "session_length_sec": 3600,
                    "time_since_last_commit_sec": 1800,
                }
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "probability" in data
        assert "confidence" in data
        assert data["confidence"] in ("weak", "moderate", "strong")

    def test_predict_no_input(self, client: TestClient) -> None:
        resp = client.post("/predict/stuck", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["probability"] == 0.5
        assert data["confidence"] == "weak"


class TestSuggestEndpoint:
    def test_predict_returns_workflow_state(self, client: TestClient) -> None:
        resp = client.post("/predict/suggest", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "flow_state" in data
        assert "dominant_state" in data
        assert "momentum" in data
        assert "focus_score" in data
        assert "method" in data
        assert "confidence" in data
        # Flow state should have all 5 states.
        from sigil_ml.models.workflow import FLOW_STATES

        for state in FLOW_STATES:
            assert state in data["flow_state"]

    def test_predict_with_classified_events(self, client: TestClient) -> None:
        from sigil_ml.models.workflow import FLOW_STATES

        events = [
            {"kind": "file", "_category": "editing", "ts": 1000},
            {"kind": "terminal", "_category": "verifying", "ts": 2000},
        ]
        resp = client.post("/predict/suggest", json={"classified_events": events})
        assert resp.status_code == 200
        data = resp.json()
        assert data["dominant_state"] in FLOW_STATES


class TestDurationEndpoint:
    def test_predict_with_features(self, client: TestClient) -> None:
        resp = client.post(
            "/predict/duration",
            json={
                "features": {
                    "file_count": 10,
                    "total_edits": 80,
                    "time_of_day_hour": 14,
                    "branch_name_length": 25,
                }
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "estimated_minutes" in data
        assert "confidence_interval" in data
        assert len(data["confidence_interval"]) == 2

    def test_predict_no_input(self, client: TestClient) -> None:
        resp = client.post("/predict/duration", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["estimated_minutes"] == 60.0


class TestTrainEndpoint:
    def test_train_returns_started(self, client: TestClient) -> None:
        resp = client.post("/train", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"

    def test_train_with_custom_db(self, client: TestClient, tmp_path) -> None:
        db = str(tmp_path / "custom.db")
        resp = client.post("/train", json={"db": db})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"
        assert "custom.db" in data["message"]
