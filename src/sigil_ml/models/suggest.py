"""Suggestion policy using Thompson Sampling (contextual bandit)."""

import json
import logging
import random
from pathlib import Path

from sigil_ml import config

logger = logging.getLogger(__name__)

ACTIONS = [
    "stay_silent",
    "suggest_commit",
    "suggest_step_back",
    "suggest_next_task",
    "suggest_test",
    "positive_reinforcement",
]


class SuggestionPolicy:
    """Thompson Sampling bandit for choosing developer suggestions."""

    def __init__(self) -> None:
        self.alphas: dict[str, float] = {a: 1.0 for a in ACTIONS}
        self.betas: dict[str, float] = {a: 1.0 for a in ACTIONS}

        weights = config.weights_path("suggest")
        if weights.exists():
            try:
                data = json.loads(weights.read_text())
                self.alphas = data.get("alphas", self.alphas)
                self.betas = data.get("betas", self.betas)
                logger.info("Loaded suggestion policy from %s", weights)
            except Exception:
                logger.warning("Failed to load suggestion policy, using priors")

    @property
    def is_trained(self) -> bool:
        """Consider trained if any arm has been updated beyond the prior."""
        return any(
            self.alphas[a] != 1.0 or self.betas[a] != 1.0 for a in ACTIONS
        )

    def predict(self, state: dict | None = None) -> dict:
        """Sample from Beta distributions to choose an action.

        Args:
            state: Optional context features (reserved for future contextual use).

        Returns:
            {"action": str, "confidence": float}
        """
        samples = {}
        for action in ACTIONS:
            samples[action] = random.betavariate(
                self.alphas[action], self.betas[action]
            )

        best_action = max(samples, key=samples.get)  # type: ignore[arg-type]
        confidence = samples[best_action]

        return {
            "action": best_action,
            "confidence": round(confidence, 4),
        }

    def update(self, action: str, reward: float) -> None:
        """Update alpha/beta for a single action based on reward.

        Args:
            action: The action that was taken.
            reward: Reward signal in [0, 1]. 1 = positive, 0 = negative.
        """
        if action not in self.alphas:
            logger.warning("Unknown action: %s", action)
            return

        self.alphas[action] += reward
        self.betas[action] += 1.0 - reward
        self._save()

    def train(self, history: list[dict]) -> None:
        """Batch update from suggestion feedback history.

        Args:
            history: List of dicts with "action" and "reward" keys.
        """
        for entry in history:
            action = entry.get("action", "")
            reward = float(entry.get("reward", 0.0))
            if action in self.alphas:
                self.alphas[action] += reward
                self.betas[action] += 1.0 - reward
        self._save()

    def _save(self) -> None:
        """Persist alpha/beta parameters to disk."""
        weights = config.weights_path("suggest")
        # The weights path has .joblib extension from config, but we store JSON.
        # Use a .json sibling instead.
        json_path = weights.with_suffix(".json")
        data = {"alphas": self.alphas, "betas": self.betas}
        json_path.write_text(json.dumps(data, indent=2))
        # Also write to the canonical path for consistency
        weights.write_text(json.dumps(data, indent=2))
        logger.info("Saved suggestion policy to %s", weights)
