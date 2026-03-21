"""FastAPI application factory and model lifecycle."""

import asyncio
import logging

from fastapi import FastAPI

from sigil_ml import config
from sigil_ml.models.activity import ActivityClassifier
from sigil_ml.models.duration import DurationEstimator
from sigil_ml.models.quality import QualityEstimator
from sigil_ml.models.stuck import StuckPredictor
from sigil_ml.models.workflow import WorkflowStatePredictor
from sigil_ml.poller import EventPoller
from sigil_ml.routes import register_routes
from sigil_ml.schema import ensure_ml_tables
from sigil_ml.training.scheduler import TrainingScheduler

logger = logging.getLogger("sigil_ml")


class AppState:
    """Holds model instances and runtime state, passed to routes."""

    def __init__(self) -> None:
        self.stuck: StuckPredictor | None = None
        self.activity: ActivityClassifier | None = None
        self.workflow: WorkflowStatePredictor | None = None
        self.duration: DurationEstimator | None = None
        self.quality: QualityEstimator | None = None
        self.poller: EventPoller | None = None
        self.training_in_progress: bool = False

    def load_models(self) -> None:
        """Load or reload all model instances from disk."""
        self.stuck = StuckPredictor()
        self.activity = ActivityClassifier()
        self.workflow = WorkflowStatePredictor()
        self.duration = DurationEstimator()
        self.quality = QualityEstimator()

    def reload_models_into_poller(self) -> None:
        """Reload model instances after retraining."""
        self.load_models()
        if self.poller:
            self.poller.stuck = self.stuck
            self.poller.activity = self.activity
            self.poller.workflow = self.workflow
            self.poller.duration = self.duration
            self.poller.quality = self.quality
        logger.info("models reloaded into poller")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(title="sigil-ml", version="0.1.0")
    state = AppState()

    register_routes(application, state)

    @application.on_event("startup")
    async def startup_event() -> None:
        db = config.db_path()

        try:
            ensure_ml_tables(db)
        except Exception:
            logger.warning("schema bootstrap failed (sigild may not have started yet)", exc_info=True)

        state.load_models()

        state.poller = EventPoller(
            db_path=db,
            models={
                "stuck": state.stuck,
                "activity": state.activity,
                "workflow": state.workflow,
                "duration": state.duration,
                "quality": state.quality,
            },
        )
        asyncio.create_task(state.poller.run())

        scheduler = TrainingScheduler(db, reload_callback=state.reload_models_into_poller)

        async def _schedule_loop():
            while True:
                await asyncio.get_event_loop().run_in_executor(None, scheduler.check_and_retrain)
                await asyncio.sleep(600)

        asyncio.create_task(_schedule_loop())

        logger.info("sigil-ml: models loaded, poller started, scheduler active")

    @application.on_event("shutdown")
    async def shutdown_event() -> None:
        if state.poller:
            state.poller.stop()
            logger.info("poller stopped")

    return application


# Module-level app instance for uvicorn import (sigil_ml.app:app).
app = create_app()
