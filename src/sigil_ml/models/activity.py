"""Activity classifier for semantic event categorization."""

from __future__ import annotations

import logging

import joblib
import numpy as np

from sigil_ml import config

logger = logging.getLogger(__name__)

CATEGORIES = [
    "editing",
    "verifying",
    "navigating",
    "researching",
    "integrating",
    "communicating",
    "idle",
]

# Full categories used after ML training splits editing into creating/refining.
CATEGORIES_FULL = [
    "creating",
    "refining",
    "verifying",
    "navigating",
    "researching",
    "integrating",
    "communicating",
    "idle",
]

# Terminal commands that indicate verifying activity.
_VERIFY_PREFIXES = (
    "go test", "go build", "go vet",
    "make", "cargo test", "cargo build",
    "npm test", "npm run test", "npm run build",
    "pytest", "python -m pytest", "python -m unittest",
    "./gradlew", "mvn test", "mvn build",
    "flake8", "pylint", "mypy", "ruff",
    "jest", "vitest", "mocha",
)

# Terminal commands that indicate integrating activity.
_INTEGRATE_PREFIXES = (
    "git commit", "git push", "git merge", "git rebase",
    "git tag", "gh pr",
)


class ActivityClassifier:
    """Classifies raw events into semantic activity categories.

    Starts rule-based on cold start, upgrades to ML (SGDClassifier)
    after sufficient training data (~500 events).

    Categories (cold start):
        editing     - file creation and modification (splits into creating/refining after ML)
        verifying   - test, build, lint commands
        navigating  - window/file/branch switches
        researching - AI queries, doc browsing
        integrating - commits, merges, pushes
        communicating - chat, review events (plugin-sourced)
        idle        - gaps between events
    """

    def __init__(self) -> None:
        self._ml_model = None
        self._trained = False

        weights = config.weights_path("activity")
        if weights.exists():
            try:
                self._ml_model = joblib.load(weights)
                self._trained = True
                logger.info("Loaded activity classifier from %s", weights)
            except Exception:
                logger.warning("Failed to load activity classifier, using rules")
                self._ml_model = None

    @property
    def is_trained(self) -> bool:
        return self._trained

    def classify(self, event: dict) -> dict:
        """Classify a single event into an activity category.

        Args:
            event: Dict with at least 'kind' key. Optionally 'payload', 'source'.

        Returns:
            {"category": str, "confidence": float, "method": "rules"|"ml"}
        """
        if self._trained and self._ml_model is not None:
            return self._classify_ml(event)
        return self._classify_rules(event)

    def classify_batch(self, events: list[dict]) -> list[dict]:
        """Classify a list of events.

        Args:
            events: List of event dicts.

        Returns:
            List of classification dicts, one per event.
        """
        return [self.classify(e) for e in events]

    def _classify_rules(self, event: dict) -> dict:
        """Deterministic rule-based classification from event kind + payload."""
        kind = event.get("kind", "")
        payload = event.get("payload") or {}
        if isinstance(payload, str):
            payload = {}

        # File events → editing (creating/refining split after ML training).
        if kind == "file":
            return {"category": "editing", "confidence": 0.8, "method": "rules"}

        # Terminal events → check command to distinguish verifying vs other.
        if kind == "terminal":
            cmd = str(payload.get("cmd", "")).strip().lower()

            # Check for verifying commands (test, build, lint).
            for prefix in _VERIFY_PREFIXES:
                if cmd.startswith(prefix):
                    return {"category": "verifying", "confidence": 0.9, "method": "rules"}

            # Check for integrating commands (git commit, push, merge).
            for prefix in _INTEGRATE_PREFIXES:
                if cmd.startswith(prefix):
                    return {"category": "integrating", "confidence": 0.9, "method": "rules"}

            # Other terminal commands → editing (likely running code).
            return {"category": "editing", "confidence": 0.6, "method": "rules"}

        # Git events → integrating.
        if kind == "git":
            return {"category": "integrating", "confidence": 0.8, "method": "rules"}

        # AI interaction events → researching.
        if kind == "ai":
            return {"category": "researching", "confidence": 0.85, "method": "rules"}

        # Window focus / compositor events → navigating.
        if kind == "hyprland":
            return {"category": "navigating", "confidence": 0.8, "method": "rules"}

        # Process events → navigating (app switching).
        if kind == "process":
            return {"category": "navigating", "confidence": 0.6, "method": "rules"}

        # Plugin-sourced events.
        source = event.get("source", "")
        if source in ("github", "jira", "slack"):
            return {"category": "communicating", "confidence": 0.7, "method": "rules"}

        # Unknown → idle.
        return {"category": "idle", "confidence": 0.5, "method": "rules"}

    def _classify_ml(self, event: dict) -> dict:
        """ML-based classification using trained SGDClassifier."""
        from sigil_ml.features import extract_activity_features

        features = extract_activity_features(event)
        feature_names = sorted(features.keys())
        x = np.array([[features[f] for f in feature_names]])

        try:
            category = self._ml_model.predict(x)[0]
            proba = self._ml_model.predict_proba(x)[0]
            confidence = float(max(proba))
        except Exception:
            logger.debug("ML classification failed, falling back to rules", exc_info=True)
            return self._classify_rules(event)

        return {"category": category, "confidence": round(confidence, 4), "method": "ml"}

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train or incrementally update the ML classifier.

        Uses SGDClassifier with partial_fit for incremental learning.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Category labels (strings from CATEGORIES or CATEGORIES_FULL).
        """
        from sklearn.linear_model import SGDClassifier

        if self._ml_model is None:
            self._ml_model = SGDClassifier(loss="log_loss", random_state=42)

        classes = np.array(CATEGORIES_FULL)
        self._ml_model.partial_fit(X, y, classes=classes)
        self._trained = True
        self._save()

    def _save(self) -> None:
        """Persist the ML model to disk."""
        if self._ml_model is not None:
            weights = config.weights_path("activity")
            joblib.dump(self._ml_model, weights)
            logger.info("Saved activity classifier to %s", weights)
