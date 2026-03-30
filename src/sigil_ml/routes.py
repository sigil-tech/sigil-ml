"""API endpoint handlers and request/response schemas."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel, Field

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
    db: str | None = Field(None, description="Deprecated: ignored, kept for backward compat")


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
        try:
            status_data = state.store.get_status_data()
            return {
                "cursor": status_data["cursor"],
                "latest_predictions": status_data["latest_predictions"],
                "poller_running": state.poller is not None and state.poller._running,
            }
        except Exception:
            return {"cursor": None, "latest_predictions": [], "poller_running": False}

    @fastapi_app.post("/predict/stuck", response_model=StuckResponse)
    async def predict_stuck(req: StuckRequest) -> StuckResponse:
        if state.stuck is None:
            return StuckResponse(probability=0.5, confidence="weak")

        if req.features is not None:
            features = req.features
        elif req.task_id is not None:
            features = extract_stuck_features(state.store, req.task_id)
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
            features = extract_duration_features(state.store, req.task_id)
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

        background_tasks.add_task(_run_training, state)
        return TrainResponse(status="started", message="Training started")


def _run_training(state: AppState) -> None:
    """Run training in a background thread."""
    try:
        state.training_in_progress = True
        trainer = Trainer(state.store)
        result = trainer.train_all()
        logger.info("Training complete: %s", result)
        state.load_models()
    except Exception:
        logger.exception("Training failed")
    finally:
        state.training_in_progress = False
