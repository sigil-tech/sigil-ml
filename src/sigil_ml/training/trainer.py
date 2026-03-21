"""Trainer: reads from the user's SQLite DB and retrains all models."""

import logging
import sqlite3
import time
from pathlib import Path

import numpy as np

from sigil_ml.features import extract_duration_features, extract_stuck_features
from sigil_ml.models.duration import FEATURE_NAMES as DURATION_FEATURES
from sigil_ml.models.duration import DurationEstimator
from sigil_ml.models.stuck import FEATURE_NAMES as STUCK_FEATURES
from sigil_ml.models.stuck import StuckPredictor
from sigil_ml.training.synthetic import generate_duration_data, generate_stuck_data

logger = logging.getLogger(__name__)


class Trainer:
    """Orchestrates training of all sigil-ml models from local data."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

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
        if not self.db_path.exists():
            logger.warning("Database not found: %s", self.db_path)
            return 0

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT id FROM tasks WHERE completed_at IS NOT NULL").fetchall()
        finally:
            conn.close()

        if len(rows) < 10:
            logger.info("Not enough completed tasks for stuck training (%d)", len(rows))
            X, y = generate_stuck_data(500)
            predictor = StuckPredictor()
            predictor.train(X, y)
            return 500

        X_list = []
        y_list = []
        for row in rows:
            task_id = row["id"]
            features = extract_stuck_features(self.db_path, task_id)
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
        if not self.db_path.exists():
            return 0

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT id, started_at, completed_at FROM tasks "
                "WHERE completed_at IS NOT NULL AND started_at IS NOT NULL"
            ).fetchall()
        finally:
            conn.close()

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
            features = extract_duration_features(self.db_path, task_id)
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
