"""Stuck predictor using GradientBoostingClassifier."""

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from sigil_ml import config

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "test_failure_count",
    "time_in_phase_sec",
    "edit_velocity",
    "file_switch_rate",
    "session_length_sec",
    "time_since_last_commit_sec",
]


class StuckPredictor:
    """Predicts whether a developer is stuck on a task."""

    def __init__(self) -> None:
        self.model: GradientBoostingClassifier | None = None
        self._trained = False
        weights = config.weights_path("stuck")
        if weights.exists():
            try:
                self.model = joblib.load(weights)
                self._trained = True
                logger.info("Loaded stuck model from %s", weights)
            except Exception:
                logger.warning("Failed to load stuck model weights, starting fresh")
                self.model = None

    @property
    def is_trained(self) -> bool:
        return self._trained

    def predict(self, features: dict) -> dict:
        """Predict stuck probability from feature dict.

        Returns:
            {"probability": float, "confidence": "weak"|"moderate"|"strong"}
        """
        if self.model is None or not self._trained:
            return {"probability": 0.5, "confidence": "weak"}

        x = np.array([[features.get(f, 0.0) for f in FEATURE_NAMES]])
        prob = float(self.model.predict_proba(x)[0, 1])

        if prob < 0.4:
            confidence = "weak"
        elif prob < 0.7:
            confidence = "moderate"
        else:
            confidence = "strong"

        return {"probability": round(prob, 4), "confidence": confidence}

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model and save weights.

        Args:
            X: Feature matrix of shape (n_samples, 6).
            y: Binary labels (0 = not stuck, 1 = stuck).
        """
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )
        self.model.fit(X, y)
        self._trained = True

        weights = config.weights_path("stuck")
        joblib.dump(self.model, weights)
        logger.info("Saved stuck model to %s", weights)
