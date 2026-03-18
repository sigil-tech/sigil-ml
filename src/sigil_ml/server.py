"""FastAPI server for sigil-ml predictions and training."""

import asyncio
import logging
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field

from sigil_ml import config
from sigil_ml.models.stuck import StuckPredictor
from sigil_ml.models.suggest import SuggestionPolicy
from sigil_ml.models.duration import DurationEstimator

logger = logging.getLogger("sigil_ml")

app = FastAPI(title="sigil-ml", version="0.1.0")

# ---------- Global model instances ----------

_stuck: StuckPredictor | None = None
_suggest: SuggestionPolicy | None = None
_duration: DurationEstimator | None = None
_training_in_progress = False


def _load_models() -> None:
    global _stuck, _suggest, _duration
    _stuck = StuckPredictor()
    _suggest = SuggestionPolicy()
    _duration = DurationEstimator()


@app.on_event("startup")
async def startup_event() -> None:
    _load_models()
    logger.info("Models loaded")


# ---------- Request / Response schemas ----------


class StuckRequest(BaseModel):
    task_id: str | None = None
    features: dict[str, float] | None = None


class StuckResponse(BaseModel):
    probability: float
    confidence: str


class SuggestRequest(BaseModel):
    task_id: str | None = None
    state: dict[str, float] | None = None


class SuggestResponse(BaseModel):
    action: str
    confidence: float


class DurationRequest(BaseModel):
    task_id: str | None = None
    features: dict[str, float] | None = None


class DurationResponse(BaseModel):
    estimated_minutes: float
    confidence_interval: list[float]


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


# ---------- Endpoints ----------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Report health and model readiness."""
    models_status = {}

    if _stuck is not None:
        models_status["stuck"] = "ready" if _stuck.is_trained else "untrained"
    else:
        models_status["stuck"] = "not_loaded"

    if _suggest is not None:
        models_status["suggest"] = "ready" if _suggest.is_trained else "untrained"
    else:
        models_status["suggest"] = "not_loaded"

    if _duration is not None:
        models_status["duration"] = "ready" if _duration.is_trained else "untrained"
    else:
        models_status["duration"] = "not_loaded"

    return HealthResponse(
        status="ok",
        models=models_status,
        uptime_sec=round(time.time() - _start_time, 1),
    )


@app.post("/predict/stuck", response_model=StuckResponse)
async def predict_stuck(req: StuckRequest) -> StuckResponse:
    """Predict whether the developer is stuck."""
    if _stuck is None:
        return StuckResponse(probability=0.5, confidence="weak")

    if req.features is not None:
        features = req.features
    elif req.task_id is not None:
        from sigil_ml.features import extract_stuck_features
        features = extract_stuck_features(config.db_path(), req.task_id)
    else:
        return StuckResponse(probability=0.5, confidence="weak")

    result = _stuck.predict(features)
    return StuckResponse(**result)


@app.post("/predict/suggest", response_model=SuggestResponse)
async def predict_suggest(req: SuggestRequest) -> SuggestResponse:
    """Get a suggestion for the developer."""
    if _suggest is None:
        return SuggestResponse(action="stay_silent", confidence=0.5)

    state = req.state
    if state is None and req.task_id is not None:
        from sigil_ml.features import extract_suggest_features
        state = extract_suggest_features(config.db_path(), req.task_id)

    result = _suggest.predict(state)
    return SuggestResponse(**result)


@app.post("/predict/duration", response_model=DurationResponse)
async def predict_duration(req: DurationRequest) -> DurationResponse:
    """Estimate task duration."""
    if _duration is None:
        return DurationResponse(estimated_minutes=60.0, confidence_interval=[30.0, 90.0])

    if req.features is not None:
        features = req.features
    elif req.task_id is not None:
        from sigil_ml.features import extract_duration_features
        features = extract_duration_features(config.db_path(), req.task_id)
    else:
        return DurationResponse(estimated_minutes=60.0, confidence_interval=[30.0, 90.0])

    result = _duration.predict(features)
    return DurationResponse(**result)


def _run_training(db_path: str) -> None:
    """Run training in a background thread."""
    global _training_in_progress, _stuck, _suggest, _duration
    try:
        _training_in_progress = True
        from sigil_ml.training.trainer import Trainer
        trainer = Trainer(db_path)
        result = trainer.train_all()
        logger.info("Training complete: %s", result)
        # Reload models after training
        _load_models()
    except Exception:
        logger.exception("Training failed")
    finally:
        _training_in_progress = False


@app.post("/train", response_model=TrainResponse)
async def train(req: TrainRequest, background_tasks: BackgroundTasks) -> TrainResponse:
    """Trigger model training (runs in background)."""
    global _training_in_progress
    if _training_in_progress:
        return TrainResponse(status="busy", message="Training already in progress")

    db = req.db or str(config.db_path())
    background_tasks.add_task(_run_training, db)
    return TrainResponse(status="started", message=f"Training started with db={db}")


# ---------- CLI entry point ----------


def main() -> None:
    """Entry point for the sigil-ml CLI."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Sigil ML sidecar")
    sub = parser.add_subparsers(dest="command")

    serve_parser = sub.add_parser("serve", help="Start the ML server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=7774)

    train_parser = sub.add_parser("train", help="Train models from local data")
    train_parser.add_argument("--db", help="Path to sigild SQLite database")

    sub.add_parser("health-check", help="Check if server is running")

    args = parser.parse_args()

    if args.command == "serve":
        uvicorn.run(
            "sigil_ml.server:app",
            host=args.host,
            port=args.port,
            log_level="info",
        )
    elif args.command == "train":
        from sigil_ml.training.trainer import Trainer
        db = args.db or str(config.db_path())
        print(f"Training models from {db} ...")
        trainer = Trainer(db)
        result = trainer.train_all()
        print(f"Done: {result}")
    elif args.command == "health-check":
        import httpx
        try:
            resp = httpx.get("http://127.0.0.1:7774/health", timeout=5)
            data = resp.json()
            print(f"Status: {data['status']}")
            for model, state in data.get("models", {}).items():
                print(f"  {model}: {state}")
        except Exception as e:
            print(f"Server not reachable: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
