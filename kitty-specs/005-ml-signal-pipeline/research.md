# Research: ML Signal Pipeline

**Feature**: 005-ml-signal-pipeline
**Date**: 2026-03-30

## R1: Poller Integration Architecture

**Decision**: Extend the existing EventPoller with signal detection hooks
**Rationale**: The poller already has the classified event buffer (200 events), activity classifications, task context, and DataStore connection. A separate SignalEngine would duplicate all of this.
**Alternatives considered**: Separate SignalEngine goroutine — rejected due to event buffer duplication and added complexity.

**Integration points identified**:
- **Per-event hook** (after ActivityClassifier.classify): Detect individual event anomalies
- **Buffer aggregate hook** (inside `_poll_once`): Detect aggregate pattern deviations using rolling window
- **Task context hook** (via `get_active_task`/`get_session_info`): Detect task-relative deviations

Signal detection runs on every poll cycle (0.5s) for event-driven signal writes. Existing predictions continue on the 60s cadence.

## R2: Profile Storage

**Decision**: Store behavior profile in `ml_predictions` with `model = 'profile'`
**Rationale**: Uses existing infrastructure. Go daemon can read it directly for LLM prompt context. No new read paths needed.
**Format**: JSON blob with tool frequencies, file type distributions, commit/test cadence, session patterns.

## R3: Pattern Detection Model

**Decision**: Hybrid — rolling z-score statistics for cold start, Isolation Forest after sufficient feedback
**Rationale**: Z-score produces interpretable evidence ("edit velocity is 2.3 std devs above your baseline") that the LLM can explain clearly. Isolation Forest catches multi-dimensional anomalies once we have labeled training data.
**Alternatives considered**:
- Pure z-score — sufficient for single-metric deviations but misses multi-dimensional patterns
- Pure Isolation Forest — less interpretable evidence, poor cold-start behavior
- LSTM/transformer — overkill for laptop constraints, GPU dependency

**Implementation**:
- Cold start: Per-metric rolling mean + stddev, signal when |z| > 2.0 (adaptive threshold)
- ML upgrade: scikit-learn `IsolationForest(n_estimators=100, contamination='auto')` after 500+ labeled events
- Evidence structure: `{metric, observed, baseline_mean, baseline_std, z_score}` for z-score; `{anomaly_score, feature_contributions}` for Isolation Forest

## R4: Next-Action Predictor

**Decision**: N-gram model on composite action tokens (category:tool)
**Rationale**: Composite tokens like `verifying:pytest`, `editing:python`, `integrating:git-commit` capture tool-specific patterns. Vocabulary is bounded by observed behavior (profile-filtered).
**Alternatives considered**: LSTM — too heavyweight for laptop. Pure category n-grams — miss tool specifics.

**Implementation**:
- Tokens derived from ActivityClassifier category + inferred tool from event payload
- N-gram order: 3-5 (configurable), with backoff to lower orders
- Prediction: probability distribution over next token. Divergence signal when actual token has < 5% predicted probability.
- Training: built incrementally from event sequences within completed tasks

## R5: File Recommender

**Decision**: Co-occurrence matrix within task sessions
**Rationale**: Simple, effective, interpretable. "Files A and B are edited together in 80% of tasks" is clear evidence.
**Implementation**:
- Build co-occurrence counts from completed tasks (file pairs edited in same task)
- Normalize to conditional probabilities: P(file B | file A being edited)
- Signal when P > threshold and file B hasn't been opened yet in current session
- Scoped to current repository (extract repo root from file paths)

## R6: Event Data Analysis

**Findings from 149k events**:
- 3,808 distinct file paths — sufficient for co-occurrence patterns
- Process events include category (ai, vcs, build, test, deploy, runtime) and command — sufficient for composite action tokens
- Git events provide commit cadence
- Terminal events (54) are sparse — rely on process events for command detection
- Event rate: ~28k events/day — profile baselines establish within 1-2 days

## R7: Training Data Requirements

| Model | Cold Start | ML Threshold | Training Source |
|-------|-----------|-------------|-----------------|
| Pattern Detector (z-score) | Immediate (rolling stats) | N/A | Event stream |
| Pattern Detector (Isolation Forest) | 500+ labeled feedback events | Suggestion accept/dismiss | ml_signals + suggestions |
| Next-Action Predictor | 1000+ events | 10+ completed tasks | Event sequences in tasks |
| File Recommender | 5+ completed tasks | Same | File edit co-occurrence |
| Behavior Profile | 100+ events | N/A | Always incremental |
