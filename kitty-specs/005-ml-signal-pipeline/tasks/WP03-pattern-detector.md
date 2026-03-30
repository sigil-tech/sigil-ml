---
work_package_id: "WP03"
title: "Pattern Detector"
lane: "planned"
dependencies: ["WP02"]
subtasks:
  - "T013"
  - "T014"
  - "T015"
  - "T016"
  - "T017"
  - "T018"
phase: "Phase 1 - Signal Models"
assignee: ""
agent: ""
shell_pid: ""
review_status: ""
reviewed_by: ""
requirement_refs:
  - "FR-003"
  - "FR-004"
  - "FR-010"
history:
  - timestamp: "2026-03-30T18:27:35Z"
    lane: "planned"
    agent: "system"
    shell_pid: ""
    action: "Prompt generated via /spec-kitty.tasks"
---

# Work Package Prompt: WP03 -- Pattern Detector

## IMPORTANT: Review Feedback Status

**Read this first if you are implementing this task!**

- **Has review feedback?**: Check the `review_status` field above. If it says `has_feedback`, scroll to the **Review Feedback** section immediately.
- **You must address all feedback** before your work is complete.
- **Mark as acknowledged**: When you understand the feedback and begin addressing it, update `review_status: acknowledged` in the frontmatter.

---

## Review Feedback

> **Populated by `/spec-kitty.review`** -- Reviewers add detailed feedback here when work needs changes.

*[This section is empty initially.]*

---

## Markdown Formatting
Wrap HTML/XML tags in backticks: `` `<div>` ``, `` `<script>` ``
Use language identifiers in code blocks: ````python`, ````bash`

---

## Implementation Command

```bash
spec-kitty implement WP03 --base WP02
```

Depends on WP02 (BehaviorProfile class provides baseline metrics).

---

## Objectives & Success Criteria

1. `PatternDetector` computes current behavioral metrics from the event buffer and compares them against the user's baseline (from `BehaviorProfile`).
2. When a metric deviates significantly (|z| > threshold), a `Signal` is emitted with structured evidence.
3. No signals are emitted when behavior is within normal range for this user.
4. No signals are emitted when fewer than 50 observations exist for a metric (cold-start safety).
5. No signals reference tools/patterns the user has never used (profile-filtered).
6. The IsolationForest upgrade path is stubbed but not active until training data (WP07) is available.

## Context & Constraints

- **Spec**: FR-003 (learns baselines, detects deviations), FR-004 (no hardcoded categories or fixed thresholds), FR-010 (cold-start rule-based).
- **Plan**: Design D3 (z-score cold start) defines the detection algorithm. Threshold default 2.0, configurable.
- **Research**: R3 confirms hybrid approach -- z-score for interpretability, IsolationForest for multi-dimensional anomaly detection after 500+ labeled events.
- **BehaviorProfile**: Provides `metrics` dict of `RollingStat` objects with mean, std, and count. Access via `profile.metrics["edit_velocity"]`.
- **Signal Dataclass**: From WP01 `src/sigil_ml/signals/__init__.py`. Produce `Signal(signal_type=..., confidence=..., evidence=...)`.
- **Existing Pattern**: Follow `ActivityClassifier` in `models/activity.py` -- cold-start rules, `is_trained` flag, ML upgrade.

---

## Subtasks & Detailed Guidance

### Subtask T013 -- Create PatternDetector Class

- **Purpose**: Define the `PatternDetector` class skeleton with the `detect()` entry point.
- **Steps**:
  1. Create `src/sigil_ml/signals/pattern_detector.py`:
     ```python
     """Pattern detection via z-score deviation from personal behavioral baselines.

     Cold start: per-metric rolling z-score with configurable threshold.
     ML upgrade: IsolationForest after 500+ labeled feedback events.
     """

     from __future__ import annotations

     import logging
     from typing import Any

     from sigil_ml.signals import Signal
     from sigil_ml.signals.profile import BehaviorProfile, RollingStat

     logger = logging.getLogger(__name__)

     # Minimum observations per metric before signals are emitted.
     MIN_OBSERVATIONS = 50
     DEFAULT_Z_THRESHOLD = 2.0


     class PatternDetector:
         """Detects behavioral deviations from personal baselines.

         Uses per-metric z-score analysis in cold-start mode.
         Upgrades to IsolationForest after sufficient training data.
         """

         def __init__(
             self,
             z_threshold: float = DEFAULT_Z_THRESHOLD,
             min_observations: int = MIN_OBSERVATIONS,
         ) -> None:
             self._z_threshold = z_threshold
             self._min_observations = min_observations
             self._isolation_forest = None
             self._is_trained = False

         @property
         def is_trained(self) -> bool:
             """True if IsolationForest model is loaded and ready."""
             return self._is_trained

         def detect(
             self, buffer: list[dict], profile: BehaviorProfile
         ) -> list[Signal]:
             """Run pattern detection on the current event buffer.

             Args:
                 buffer: Recent classified events from the poller.
                 profile: Current user behavior profile.

             Returns:
                 List of Signal objects for detected deviations.
             """
             if self._is_trained and self._isolation_forest is not None:
                 return self._detect_ml(buffer, profile)
             return self._detect_zscore(buffer, profile)
     ```
  2. Export from `src/sigil_ml/signals/__init__.py`:
     ```python
     from sigil_ml.signals.pattern_detector import PatternDetector
     ```
- **Files**: `src/sigil_ml/signals/pattern_detector.py` (new), `src/sigil_ml/signals/__init__.py` (update)
- **Parallel?**: No -- T015-T018 build on this skeleton.
- **Validation**:
  - [ ] `PatternDetector()` constructs without errors
  - [ ] `is_trained` returns False by default
  - [ ] `detect([], profile)` returns empty list

### Subtask T014 -- Implement RollingStat Helper

- **Purpose**: The `RollingStat` helper was defined in WP02 T007 inside `profile.py`. This subtask verifies it works correctly for pattern detection use cases.
- **Steps**:
  1. `RollingStat` is already defined in `src/sigil_ml/signals/profile.py` (from WP02).
  2. Verify the following properties are correct:
     - `std` property returns `variance ** 0.5`
     - `update()` maintains correct exponentially-weighted running mean and variance
     - `count` tracks total observations
     - `to_dict()` and `from_dict()` round-trip correctly
  3. If additional methods are needed for pattern detection, add them:
     ```python
     def z_score(self, value: float) -> float | None:
         """Compute z-score of value against this stat's baseline.

         Returns None if insufficient observations (count < threshold).
         """
         if self.count < 2 or self.std < 1e-10:
             return None
         return (value - self.mean) / self.std
     ```
- **Files**: `src/sigil_ml/signals/profile.py` (minor addition)
- **Parallel?**: Yes -- standalone utility, can proceed independently.
- **Validation**:
  - [ ] After 100 updates with mean ~5.0 and std ~1.0, `z_score(8.0)` returns approximately 3.0
  - [ ] `z_score()` returns None when count < 2
  - [ ] `z_score()` returns None when std is near zero (constant values)

### Subtask T015 -- Z-Score Deviation Detection

- **Purpose**: Implement the core detection loop that computes current metrics from the event buffer and checks for z-score deviations against the profile baselines.
- **Steps**:
  1. Implement `_detect_zscore()` in `PatternDetector`:
     ```python
     def _detect_zscore(
         self, buffer: list[dict], profile: BehaviorProfile
     ) -> list[Signal]:
         """Detect deviations using per-metric z-scores."""
         signals: list[Signal] = []
         current_metrics = self._compute_current_metrics(buffer)

         for name, value in current_metrics.items():
             stat = profile.metrics.get(name)
             if stat is None or stat.count < self._min_observations:
                 continue  # Not enough baseline data for this metric

             z = stat.z_score(value)
             if z is None:
                 continue

             if abs(z) > self._z_threshold:
                 confidence = min(abs(z) / 4.0, 0.95)
                 signals.append(Signal(
                     signal_type=f"{name}_deviation",
                     confidence=round(confidence, 4),
                     evidence=self._build_evidence(name, value, stat, z),
                     suggested_action=self._infer_action(name, z),
                 ))

         return signals
     ```
  2. Implement `_compute_current_metrics()`:
     ```python
     def _compute_current_metrics(self, buffer: list[dict]) -> dict[str, float]:
         """Compute behavioral metrics from the current event buffer."""
         if not buffer:
             return {}

         metrics: dict[str, float] = {}
         now_ms = buffer[-1].get("ts", 0)
         first_ts = buffer[0].get("ts", now_ms)
         window_min = max((now_ms - first_ts) / 60000.0, 1 / 60.0)

         # Edit velocity: file events per minute
         edit_count = sum(1 for e in buffer if e.get("kind") == "file")
         if edit_count > 0:
             metrics["edit_velocity"] = edit_count / window_min

         # Context switch rate: category transitions
         transitions = 0
         for i in range(1, len(buffer)):
             if buffer[i].get("_category") != buffer[i - 1].get("_category"):
                 transitions += 1
         if len(buffer) > 1:
             metrics["context_switch_rate"] = transitions / (len(buffer) - 1)

         # File focus: 1 - (unique_files / total_edits)
         edit_events = [e for e in buffer if e.get("kind") == "file"]
         if edit_events:
             files = set()
             for e in edit_events:
                 p = e.get("payload") or {}
                 if isinstance(p, dict) and "path" in p:
                     files.add(p["path"])
             metrics["file_focus"] = 1.0 - (len(files) / max(len(edit_events), 1))

         return metrics
     ```
  3. Metric names in `_compute_current_metrics()` must match the metric names used in `BehaviorProfile._update_rhythm_stats()` from WP02 (e.g., `edit_velocity`, `context_switch_rate`).
- **Files**: `src/sigil_ml/signals/pattern_detector.py`
- **Parallel?**: Depends on T013 skeleton.
- **Validation**:
  - [ ] Buffer with high edit velocity (3x baseline) produces a `velocity_deviation` signal
  - [ ] Buffer with normal edit velocity produces no signal
  - [ ] Metric with count < 50 in profile produces no signal (cold-start safety)
  - [ ] Confidence is bounded between 0.0 and 0.95

### Subtask T016 -- Signal Evidence Generation

- **Purpose**: Build structured evidence JSON for each detected deviation, following the data-model.md envelope format.
- **Steps**:
  1. Implement `_build_evidence()` and `_infer_action()`:
     ```python
     def _build_evidence(
         self, metric: str, observed: float, stat: RollingStat, z: float
     ) -> dict[str, Any]:
         """Build structured evidence dict for LLM rendering."""
         return {
             "source_model": "pattern_detector",
             "metric": metric,
             "observed": round(observed, 4),
             "baseline_mean": round(stat.mean, 4),
             "baseline_std": round(stat.std, 4),
             "z_score": round(z, 4),
             "observation_count": stat.count,
         }

     def _infer_action(self, metric: str, z: float) -> str | None:
         """Infer a generic suggested action from the metric and direction."""
         action_map = {
             "edit_velocity": "take_break" if z > 0 else None,
             "test_cadence": "test" if z > 0 else None,
             "commit_cadence": "commit" if z > 0 else None,
             "context_switch_rate": "investigate" if z > 0 else None,
             "file_focus": "investigate",
         }
         return action_map.get(metric)
     ```
  2. Evidence JSON must include `source_model` key per data-model.md convention.
  3. The `suggested_action` is a generic hint (not a specific command). The LLM in the Go daemon interprets it contextually.
- **Files**: `src/sigil_ml/signals/pattern_detector.py`
- **Parallel?**: Depends on T015.
- **Validation**:
  - [ ] Evidence dict contains all required keys: source_model, metric, observed, baseline_mean, baseline_std, z_score
  - [ ] `source_model` is always `"pattern_detector"`
  - [ ] `_infer_action()` returns None for unknown metrics (safe default)

### Subtask T017 -- Cold-Start Rule-Based Detection

- **Purpose**: Ensure the system behaves correctly when insufficient data exists -- no ML signals are emitted until minimum data thresholds are met.
- **Steps**:
  1. The cold-start behavior is already handled by the `count < self._min_observations` check in `_detect_zscore()` (T015).
  2. Add explicit documentation and a helper method:
     ```python
     def has_sufficient_data(self, profile: BehaviorProfile) -> bool:
         """Check if the profile has enough data for meaningful detection.

         Returns True if at least one metric has sufficient observations.
         """
         return any(
             stat.count >= self._min_observations
             for stat in profile.metrics.values()
         )
     ```
  3. Add logging in `detect()`:
     ```python
     def detect(self, buffer, profile):
         if not self.has_sufficient_data(profile):
             logger.debug("pattern_detector: insufficient baseline data, skipping")
             return []
         # ... rest of detection logic
     ```
  4. This matches spec edge case: "What happens during cold start with no event history? Rule-based fallback signals only, with low confidence. No ML signals emitted until minimum data threshold is met."
- **Files**: `src/sigil_ml/signals/pattern_detector.py`
- **Parallel?**: Depends on T015.
- **Validation**:
  - [ ] `detect()` returns empty list when profile has < 50 observations per metric
  - [ ] `has_sufficient_data()` returns False for a fresh profile
  - [ ] `has_sufficient_data()` returns True after 50+ observations on at least one metric
  - [ ] Debug logging confirms cold-start skip (visible at DEBUG level)

### Subtask T018 -- Stub IsolationForest Upgrade Path

- **Purpose**: Stub the ML upgrade path so the PatternDetector can be trained after sufficient feedback data accumulates.
- **Steps**:
  1. Add training and loading methods:
     ```python
     def train(self, feature_matrix, labels) -> None:
         """Train IsolationForest from feedback-labeled behavioral vectors.

         Called by Trainer (WP07) after 500+ labeled feedback events.

         Args:
             feature_matrix: numpy array of behavioral feature vectors.
             labels: numpy array where 1 = normal (accepted signal),
                     0 = anomalous (dismissed signal).
         """
         from sklearn.ensemble import IsolationForest
         self._isolation_forest = IsolationForest(
             n_estimators=100,
             contamination="auto",
             random_state=42,
         )
         self._isolation_forest.fit(feature_matrix)
         self._is_trained = True
         logger.info("PatternDetector: IsolationForest trained with %d samples", len(feature_matrix))

     def save(self, model_store) -> None:
         """Persist trained IsolationForest via ModelStore."""
         if self._isolation_forest is None:
             return
         import io
         import joblib
         buf = io.BytesIO()
         joblib.dump(self._isolation_forest, buf)
         model_store.save("pattern_detector", buf.getvalue())

     def load(self, model_store) -> bool:
         """Load trained IsolationForest from ModelStore. Returns True if loaded."""
         data = model_store.load("pattern_detector")
         if data is None:
             return False
         import io
         import joblib
         self._isolation_forest = joblib.load(io.BytesIO(data))
         self._is_trained = True
         return True

     def _detect_ml(self, buffer, profile):
         """ML-based detection using IsolationForest (stub).

         Active only after train() is called with sufficient data.
         Falls back to z-score if ML detection fails.
         """
         try:
             # TODO: Extract feature vector from buffer, run isolation forest
             # For now, fall back to z-score
             return self._detect_zscore(buffer, profile)
         except Exception:
             logger.debug("PatternDetector: ML detection failed, falling back to z-score")
             return self._detect_zscore(buffer, profile)
     ```
  2. The `_detect_ml()` method is intentionally a stub that falls back to z-score. Full implementation comes with WP07 (training pipeline).
  3. Follow the `ActivityClassifier` pattern: `is_trained` flag, `train()` method, `save()`/`load()` via ModelStore.
- **Files**: `src/sigil_ml/signals/pattern_detector.py`
- **Parallel?**: Yes -- independent of z-score implementation.
- **Notes**: IsolationForest from scikit-learn is already an existing dependency (`sklearn` is in the project's dependencies).
- **Validation**:
  - [ ] `train()` sets `is_trained` to True
  - [ ] `save()` and `load()` round-trip via ModelStore
  - [ ] `_detect_ml()` falls back to z-score gracefully
  - [ ] `detect()` calls `_detect_ml()` when `is_trained` is True

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Z-score false positives from non-normal distributions | Medium | Medium | Conservative threshold (2.0); confidence scaling dampens borderline signals |
| Metric name mismatch between profile and detector | Medium | High | Both use the same string constants; add a shared constant module if needed |
| IsolationForest import overhead | Low | Low | Lazy import only when train() is called; scikit-learn already loaded |
| Detection runs on every poll cycle (0.5s) | Expected | Medium | _compute_current_metrics is O(n) on buffer size (200 max); target <50ms |

## Review Guidance

- **Metric name alignment**: Verify that metric names in `_compute_current_metrics()` match the metric names in `BehaviorProfile._update_rhythm_stats()`. Any mismatch means z-scores will never be computed for that metric.
- **Cold-start safety**: Verify that no signals are emitted when `stat.count < MIN_OBSERVATIONS` (50).
- **Evidence completeness**: Verify evidence dict matches data-model.md pattern_detector evidence structure.
- **Profile filtering**: Verify that signals only reference metrics that exist in the profile (no signals for tools the user hasn't used).
- **Performance**: `_compute_current_metrics()` should not query the database -- it operates only on the in-memory buffer.

---

## Activity Log

- 2026-03-30T18:27:35Z -- system -- lane=planned -- Prompt generated via /spec-kitty.tasks.
