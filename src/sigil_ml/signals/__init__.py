"""ML Signal Pipeline — event-driven behavioral signal detection.

This package contains:
  - Signal: shared dataclass for all signal models
  - SignalEngine: orchestrator that runs all models per poll cycle (WP06)
  - BehaviorProfile: incremental per-user behavioral profile (WP02)
  - PatternDetector: z-score and Isolation Forest anomaly detection (WP03)
  - NextActionPredictor: n-gram action sequence prediction (WP04)
  - FileRecommender: co-occurrence file recommendation (WP05)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


from sigil_ml.signals.profile import BehaviorProfile, RollingStat

__all__ = [
    "BehaviorProfile",
    "RollingStat",
    "Signal",
]


@dataclass
class Signal:
    """A structured signal emitted when a model detects something noteworthy.

    Produced by signal models (PatternDetector, NextActionPredictor, FileRecommender).
    Consumed by SignalEngine, which writes it to ml_signals via DataStore.
    """

    signal_type: str
    """Model-generated type, e.g. 'velocity_deviation', 'divergence_test', 'file_cluster'."""

    confidence: float
    """Model's confidence (0.0 to 1.0) that this signal is worth surfacing."""

    evidence: dict[str, Any]
    """Structured evidence for LLM rendering. Must include 'source_model' key."""

    suggested_action: str | None = None
    """Generic action hint for LLM: 'investigate', 'test', 'commit', 'take_break'."""

    ttl_sec: int | None = None
    """Time-to-live in seconds. None = no expiry."""

    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    """Unix ms timestamp. Auto-populated on creation."""
