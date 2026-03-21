"""API endpoint handlers and request/response schemas."""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import TYPE_CHECKING

from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel, Field

from sigil_ml import config
from sigil_ml.features import extract_duration_features, extract_stuck_features
from sigil_ml.plugins import fetch_capabilities
from sigil_ml.training.trainer import Trainer

if TYPE_CHECKING:
    from sigil_ml.app import AppState

logger = logging.getLogger("sigil_ml")


# ---------- Request / Response schemas ----------


class StuckRequest(BaseModel):
    task_id: str | None = None
    features: dict[str, float] | None = None


class StuckResponse(BaseModel):
    probability: float
    confidence: str


class WorkflowStateRequest(BaseModel):
    task_id: str | None = None
    classified_events: list[dict] | None = None


class WorkflowStateResponse(BaseModel):
    flow_state: dict[str, float]
    dominant_state: str
    momentum: float
    focus_score: float
    dominant_activity: str
    activity_distribution: dict[str, float]
    session_elapsed_min: float
    method: str
    confidence: float


class DurationRequest(BaseModel):
    task_id: str | None = None
    features: dict[str, float] | None = None


class DurationResponse(BaseModel):
    estimated_minutes: float
    confidence_interval: list[float]


class QualityRequest(BaseModel):
    features: dict[str, float]


class QualityResponse(BaseModel):
    score: int
    components: dict[str, float]
    status: str


class TrainRequest(BaseModel):
    db: str | None = Field(None, description="Override path to SQLite database")


class TrainResponse(BaseModel):
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    models: dict[str, str]
    uptime_sec: float


_start_time = time.time()


# ---------- Route registration ----------


def register_routes(fastapi_app: FastAPI, state: AppState) -> None:
    """Register all API routes on the given FastAPI app."""

    @fastapi_app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        models_status = {}
        for name, model in [
            ("stuck", state.stuck),
            ("activity", state.activity),
            ("workflow", state.workflow),
            ("duration", state.duration),
        ]:
            if model is not None:
                models_status[name] = "ready" if model.is_trained else "untrained"
            else:
                models_status[name] = "not_loaded"

        models_status["quality"] = "ready" if state.quality is not None else "not_loaded"

        return HealthResponse(
            status="ok",
            models=models_status,
            uptime_sec=round(time.time() - _start_time, 1),
        )

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
                    "WHERE expires_at IS NULL OR expires_at > ? "
                    "ORDER BY created_at DESC",
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

    @fastapi_app.post("/predict/stuck", response_model=StuckResponse)
    async def predict_stuck(req: StuckRequest) -> StuckResponse:
        if state.stuck is None:
            return StuckResponse(probability=0.5, confidence="weak")

        if req.features is not None:
            features = req.features
        elif req.task_id is not None:
            features = extract_stuck_features(config.db_path(), req.task_id)
        else:
            return StuckResponse(probability=0.5, confidence="weak")

        result = state.stuck.predict(features)
        return StuckResponse(**result)

    @fastapi_app.post("/predict/suggest", response_model=WorkflowStateResponse)
    async def predict_suggest(req: WorkflowStateRequest) -> WorkflowStateResponse:
        if state.workflow is None:
            return WorkflowStateResponse(
                flow_state={
                    "shallow_work": 1.0,
                    "deep_work": 0.0,
                    "exploring": 0.0,
                    "blocked": 0.0,
                    "winding_down": 0.0,
                },
                dominant_state="shallow_work",
                momentum=0.0,
                focus_score=0.5,
                dominant_activity="idle",
                activity_distribution={},
                session_elapsed_min=0.0,
                method="rules",
                confidence=0.5,
            )

        classified_events = req.classified_events or []
        session_info = {"session_elapsed_min": 0.0, "task_phase": None, "test_failures": 0}

        if not classified_events and state.poller:
            classified_events = state.poller._buffer

        result = state.workflow.predict(classified_events, session_info)
        return WorkflowStateResponse(**result)

    @fastapi_app.post("/predict/duration", response_model=DurationResponse)
    async def predict_duration(req: DurationRequest) -> DurationResponse:
        if state.duration is None:
            return DurationResponse(estimated_minutes=60.0, confidence_interval=[30.0, 90.0])

        if req.features is not None:
            features = req.features
        elif req.task_id is not None:
            features = extract_duration_features(config.db_path(), req.task_id)
        else:
            return DurationResponse(estimated_minutes=60.0, confidence_interval=[30.0, 90.0])

        result = state.duration.predict(features)
        return DurationResponse(**result)

    @fastapi_app.post("/predict/quality", response_model=QualityResponse)
    async def predict_quality(req: QualityRequest) -> QualityResponse:
        if state.quality is None:
            return QualityResponse(score=50, components={}, status="normal")

        result = state.quality.predict(req.features)
        return QualityResponse(
            score=result["score"],
            components=result["components"],
            status=result["status"],
        )

    @fastapi_app.get("/plugins")
    async def plugins() -> dict:
        return fetch_capabilities()

    @fastapi_app.post("/train", response_model=TrainResponse)
    async def train(req: TrainRequest, background_tasks: BackgroundTasks) -> TrainResponse:
        if state.training_in_progress:
            return TrainResponse(status="busy", message="Training already in progress")

        db = req.db or str(config.db_path())
        background_tasks.add_task(_run_training, state, db)
        return TrainResponse(status="started", message=f"Training started with db={db}")


def _run_training(state: AppState, db_path: str) -> None:
    """Run training in a background thread."""
    try:
        state.training_in_progress = True
        trainer = Trainer(db_path)
        result = trainer.train_all()
        logger.info("Training complete: %s", result)
        state.load_models()
    except Exception:
        logger.exception("Training failed")
    finally:
        state.training_in_progress = False
