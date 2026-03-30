"""Trainer: reads from the DataStore and retrains all models."""

import logging
import time

import numpy as np

from sigil_ml.features import extract_duration_features, extract_stuck_features
from sigil_ml.models.duration import FEATURE_NAMES as DURATION_FEATURES
from sigil_ml.models.duration import DurationEstimator
from sigil_ml.models.stuck import FEATURE_NAMES as STUCK_FEATURES
from sigil_ml.models.stuck import StuckPredictor
from sigil_ml.store import DataStore
from sigil_ml.training.synthetic import generate_duration_data, generate_stuck_data

logger = logging.getLogger(__name__)


class Trainer:
    """Orchestrates training of all sigil-ml models from local data."""

    def __init__(self, store: DataStore) -> None:
        self.store = store

    def train_all(self) -> dict:
        """Train all models and return a summary.

        Returns:
            {"trained": [model_names], "samples": int, "duration_sec": float}
        """
        start = time.time()
        trained = []
        total_samples = 0

        stuck_result = self._train_stuck()
        if stuck_result:
            trained.append("stuck")
            total_samples += stuck_result

        duration_result = self._train_duration()
        if duration_result:
            trained.append("duration")
            total_samples += duration_result

        elapsed = time.time() - start
        return {
            "trained": trained,
            "samples": total_samples,
            "duration_sec": round(elapsed, 2),
        }

    def _train_stuck(self) -> int:
        """Train the stuck predictor from completed tasks.

        Returns:
            Number of samples used, or 0 if insufficient data.
        """
        task_ids = self.store.get_completed_task_ids()

        if len(task_ids) < 10:
            logger.info("Not enough completed tasks for stuck training (%d)", len(task_ids))
            X, y = generate_stuck_data(500)
            predictor = StuckPredictor()
            predictor.train(X, y)
            return 500

        X_list = []
        y_list = []
        for task_id in task_ids:
            features = extract_stuck_features(self.store, task_id)
            x = [features.get(f, 0.0) for f in STUCK_FEATURES]
            X_list.append(x)
            # Heuristic label: stuck if high test failures and long time in phase
            stuck = features["test_failure_count"] > 3 and features["time_in_phase_sec"] > 600
            y_list.append(1.0 if stuck else 0.0)

        X = np.array(X_list)
        y = np.array(y_list)

        predictor = StuckPredictor()
        predictor.train(X, y)
        return len(X)

    def _train_duration(self) -> int:
        """Train the duration estimator from completed tasks.

        Returns:
            Number of samples used, or 0 if insufficient data.
        """
        rows = self.store.get_completed_tasks_with_timestamps()

        if len(rows) < 10:
            logger.info("Not enough completed tasks for duration training (%d)", len(rows))
            X, y = generate_duration_data(500)
            estimator = DurationEstimator()
            estimator.train(X, y)
            return 500

        X_list = []
        y_list = []
        for row in rows:
            task_id = row["id"]
            features = extract_duration_features(self.store, task_id)
            x = [features.get(f, 0.0) for f in DURATION_FEATURES]
            X_list.append(x)
            # Duration in minutes
            duration_min = (row["completed_at"] - row["started_at"]) / 60000.0
            y_list.append(max(duration_min, 1.0))

        X = np.array(X_list)
        y = np.array(y_list)

        estimator = DurationEstimator()
        estimator.train(X, y)
        return len(X)
