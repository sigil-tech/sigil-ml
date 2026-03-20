"""Tests for ML model classes."""

import os
import tempfile

import numpy as np
import pytest

from sigil_ml.training.synthetic import generate_stuck_data, generate_duration_data


@pytest.fixture(autouse=True)
def _isolate_models(tmp_path, monkeypatch):
    """Redirect model weights to a temp directory so tests don't pollute real config."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))


class TestStuckPredictor:
    def test_untrained_returns_default(self) -> None:
        from sigil_ml.models.stuck import StuckPredictor
        model = StuckPredictor()
        result = model.predict({
            "test_failure_count": 5,
            "time_in_phase_sec": 1200,
            "edit_velocity": 4.0,
            "file_switch_rate": 0.7,
            "session_length_sec": 3600,
            "time_since_last_commit_sec": 1800,
        })
        assert result["probability"] == 0.5
        assert result["confidence"] == "weak"

    def test_train_and_predict(self) -> None:
        from sigil_ml.models.stuck import StuckPredictor
        X, y = generate_stuck_data(200)
        model = StuckPredictor()
        model.train(X, y)

        assert model.is_trained

        # Stuck-looking sample
        stuck_features = {
            "test_failure_count": 8,
            "time_in_phase_sec": 2400,
            "edit_velocity": 5.0,
            "file_switch_rate": 0.8,
            "session_length_sec": 5000,
            "time_since_last_commit_sec": 2500,
        }
        result = model.predict(stuck_features)
        assert result["probability"] > 0.5

        # Not-stuck sample
        ok_features = {
            "test_failure_count": 0,
            "time_in_phase_sec": 100,
            "edit_velocity": 1.0,
            "file_switch_rate": 0.2,
            "session_length_sec": 600,
            "time_since_last_commit_sec": 120,
        }
        result_ok = model.predict(ok_features)
        assert result_ok["probability"] < 0.5

    def test_confidence_levels(self) -> None:
        from sigil_ml.models.stuck import StuckPredictor
        X, y = generate_stuck_data(200)
        model = StuckPredictor()
        model.train(X, y)

        # The model should produce varying confidence levels across different inputs
        results = []
        for _ in range(50):
            features = {
                "test_failure_count": float(np.random.randint(0, 10)),
                "time_in_phase_sec": float(np.random.uniform(30, 3600)),
                "edit_velocity": float(np.random.uniform(0.5, 8)),
                "file_switch_rate": float(np.random.uniform(0.1, 1.0)),
                "session_length_sec": float(np.random.uniform(300, 7200)),
                "time_since_last_commit_sec": float(np.random.uniform(60, 3600)),
            }
            results.append(model.predict(features)["confidence"])
        # Should see at least 2 different confidence levels
        assert len(set(results)) >= 2

    def test_weights_persist(self) -> None:
        from sigil_ml.models.stuck import StuckPredictor
        X, y = generate_stuck_data(100)

        model1 = StuckPredictor()
        model1.train(X, y)

        features = {
            "test_failure_count": 5,
            "time_in_phase_sec": 1500,
            "edit_velocity": 4.0,
            "file_switch_rate": 0.6,
            "session_length_sec": 3000,
            "time_since_last_commit_sec": 1800,
        }
        pred1 = model1.predict(features)

        # Load from disk
        model2 = StuckPredictor()
        assert model2.is_trained
        pred2 = model2.predict(features)
        assert pred1["probability"] == pred2["probability"]


class TestSuggestionPolicy:
    def test_untrained_predict(self) -> None:
        from sigil_ml.models.suggest import SuggestionPolicy
        policy = SuggestionPolicy()
        result = policy.predict({})
        from sigil_ml.models.suggest import ACTIONS
        assert result["action"] in ACTIONS
        assert 0.0 <= result["confidence"] <= 1.0

    def test_update_shifts_distribution(self) -> None:
        from sigil_ml.models.suggest import SuggestionPolicy
        policy = SuggestionPolicy()

        # Heavily reward "suggest_commit"
        for _ in range(50):
            policy.update("suggest_commit", 1.0)

        # After many positive rewards, suggest_commit should be favored
        action_counts: dict[str, int] = {}
        for _ in range(100):
            result = policy.predict({})
            action_counts[result["action"]] = action_counts.get(result["action"], 0) + 1

        assert action_counts.get("suggest_commit", 0) > 50

    def test_train_batch(self) -> None:
        from sigil_ml.models.suggest import SuggestionPolicy
        policy = SuggestionPolicy()
        history = [
            {"action": "suggest_test", "reward": 1.0},
            {"action": "suggest_test", "reward": 1.0},
            {"action": "suggest_test", "reward": 1.0},
            {"action": "stay_silent", "reward": 0.0},
            {"action": "stay_silent", "reward": 0.0},
        ]
        policy.train(history)
        assert policy.alphas["suggest_test"] > policy.alphas["stay_silent"]


class TestDurationEstimator:
    def test_untrained_returns_default(self) -> None:
        from sigil_ml.models.duration import DurationEstimator
        model = DurationEstimator()
        result = model.predict({
            "file_count": 5,
            "total_edits": 50,
            "time_of_day_hour": 14,
            "branch_name_length": 20,
        })
        assert result["estimated_minutes"] == 60.0
        assert len(result["confidence_interval"]) == 2

    def test_train_and_predict(self) -> None:
        from sigil_ml.models.duration import DurationEstimator
        X, y = generate_duration_data(200)
        model = DurationEstimator()
        model.train(X, y)

        assert model.is_trained

        # Many files + edits should predict longer duration
        big_task = {
            "file_count": 25,
            "total_edits": 180,
            "time_of_day_hour": 14,
            "branch_name_length": 40,
        }
        result_big = model.predict(big_task)

        small_task = {
            "file_count": 2,
            "total_edits": 10,
            "time_of_day_hour": 14,
            "branch_name_length": 10,
        }
        result_small = model.predict(small_task)

        assert result_big["estimated_minutes"] > result_small["estimated_minutes"]

    def test_confidence_interval(self) -> None:
        from sigil_ml.models.duration import DurationEstimator
        X, y = generate_duration_data(200)
        model = DurationEstimator()
        model.train(X, y)

        result = model.predict({
            "file_count": 10,
            "total_edits": 80,
            "time_of_day_hour": 10,
            "branch_name_length": 25,
        })
        low, high = result["confidence_interval"]
        assert low <= result["estimated_minutes"] <= high
        assert low >= 0

    def test_weights_persist(self) -> None:
        from sigil_ml.models.duration import DurationEstimator
        X, y = generate_duration_data(100)

        model1 = DurationEstimator()
        model1.train(X, y)

        features = {
            "file_count": 10,
            "total_edits": 60,
            "time_of_day_hour": 12,
            "branch_name_length": 20,
        }
        pred1 = model1.predict(features)

        model2 = DurationEstimator()
        assert model2.is_trained
        pred2 = model2.predict(features)
        assert pred1["estimated_minutes"] == pred2["estimated_minutes"]
