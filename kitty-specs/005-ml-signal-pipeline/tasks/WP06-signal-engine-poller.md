---
work_package_id: "WP06"
title: "SignalEngine & Poller Integration"
lane: "planned"
dependencies: ["WP03", "WP04", "WP05"]
subtasks:
  - "T030"
  - "T031"
  - "T032"
  - "T033"
  - "T034"
  - "T035"
phase: "Phase 2 - Integration"
assignee: ""
agent: ""
shell_pid: ""
review_status: ""
reviewed_by: ""
requirement_refs:
  - "FR-007"
  - "FR-008"
  - "FR-009"
  - "FR-016"
history:
  - timestamp: "2026-03-30T18:27:35Z"
    lane: "planned"
    agent: "system"
    shell_pid: ""
    action: "Prompt generated via /spec-kitty.tasks"
---

# Work Package Prompt: WP06 -- SignalEngine & Poller Integration

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
spec-kitty implement WP06 --base WP03,WP04,WP05
```

Depends on WP03, WP04, WP05 (all three signal models must exist and be individually testable).

---

## Objectives & Success Criteria

1. `SignalEngine` orchestrates all three signal models on every poll cycle.
2. The engine is integrated into `EventPoller._poll_once()` -- signal detection runs on every 0.5s poll cycle (separate from the 60s prediction cadence).
3. Rate limiting prevents signal floods: max 1 per signal_type per 5 minutes, max 10 total per 5 minutes.
4. Dismissed signal cooldown suppresses a signal_type for 30 minutes after dismissal.
5. SignalEngine is initialized in `app.py` during local-mode startup and passed to the poller.
6. BehaviorProfile is written to `ml_predictions` with `model='profile'` on each prediction cycle.
7. All existing predictions (stuck, suggest, duration, activity, quality) remain unchanged -- zero regression.

## Context & Constraints

- **Spec**: FR-007 (event-driven signal writes), FR-008 (evidence, confidence, timestamps), FR-009 (profile-filtered), FR-016 (existing behavior unchanged).
- **Plan**: Design D1 (SignalEngine orchestrator), D7 (rate limiting).
- **Poller**: `src/sigil_ml/poller.py` -- `_poll_once()` runs every 0.5s. The classified event buffer is `self._buffer` (last 200 events). Prediction cycle runs every 60s.
- **App Factory**: `src/sigil_ml/app.py` -- `create_app()` initializes models and the poller in local mode startup.
- **Critical Constraint**: Signal detection is ADDITIVE. It must not modify, slow down, or interfere with the existing `_predict_and_write()` path. If signal detection fails, the prediction cycle must continue normally.

---

## Subtasks & Detailed Guidance

### Subtask T030 -- Create SignalEngine Orchestrator

- **Purpose**: Create the `SignalEngine` class that coordinates all three signal models and manages rate limiting/cooldown state.
- **Steps**:
  1. Create `src/sigil_ml/signals/engine.py`:
     ```python
     """SignalEngine: orchestrates all signal models on each poll cycle.

     Called by the EventPoller on every poll cycle. Runs the BehaviorProfile
     update, PatternDetector, NextActionPredictor, and FileRecommender.
     Writes signals to ml_signals via the DataStore.
     """

     from __future__ import annotations

     import logging
     import time
     from typing import Any

     from sigil_ml.signals import Signal
     from sigil_ml.signals.file_recommender import FileRecommender
     from sigil_ml.signals.next_action import NextActionPredictor
     from sigil_ml.signals.pattern_detector import PatternDetector
     from sigil_ml.signals.profile import BehaviorProfile
     from sigil_ml.store import DataStore

     logger = logging.getLogger(__name__)

     # Rate limiting constants
     SIGNAL_TYPE_COOLDOWN_SEC = 300   # 5 minutes per signal type
     SIGNAL_TOTAL_WINDOW_SEC = 300    # 5-minute window for total count
     SIGNAL_TOTAL_MAX = 10            # Max 10 signals per window
     DISMISSED_COOLDOWN_SEC = 1800    # 30-minute cooldown after dismissal


     class SignalEngine:
         """Orchestrates signal detection across all models.

         Called on every poll cycle (0.5s). Runs profile updates and
         all signal models. Writes emitted signals to the DataStore.
         """

         def __init__(
             self,
             store: DataStore,
             profile: BehaviorProfile,
             pattern_detector: PatternDetector,
             next_action: NextActionPredictor,
             file_recommender: FileRecommender,
         ) -> None:
             self.store = store
             self.profile = profile
             self.pattern_detector = pattern_detector
             self.next_action = next_action
             self.file_recommender = file_recommender
             # Rate limiting state
             self._recent_signals: list[tuple[str, float]] = []  # (signal_type, timestamp)
             self._dismissed_types: dict[str, float] = {}  # signal_type -> dismissed_at

         def process_events(
             self,
             buffer: list[dict],
             task_context: dict | None = None,
         ) -> int:
             """Run all signal models on the current event buffer.

             Called by the poller on every poll cycle. Writes signals
             immediately to the DataStore.

             Args:
                 buffer: Classified event buffer (last 200 events).
                 task_context: Active task info (task_id, repo_root, etc.).

             Returns:
                 Number of signals written.
             """
             if not buffer:
                 return 0

             try:
                 return self._process_events_inner(buffer, task_context)
             except Exception:
                 logger.debug("signal_engine: error in signal processing", exc_info=True)
                 return 0

         def _process_events_inner(
             self, buffer: list[dict], task_context: dict | None
         ) -> int:
             """Inner processing loop -- separated for error isolation."""
             # 1. Update behavior profile
             self.profile.update(buffer[-20:])  # Only new events since last cycle

             # 2. Update n-gram model incrementally
             self.next_action.train_incremental(
                 self.next_action._extract_tokens(buffer[-20:])
             )

             # 3. Collect signals from all models
             signals: list[Signal] = []
             signals.extend(self.pattern_detector.detect(buffer, self.profile))
             signals.extend(self.next_action.check_divergence(buffer, self.profile))
             signals.extend(
                 self.file_recommender.check(buffer, task_context, self.profile)
             )

             # 4. Apply rate limiting and cooldown
             filtered = self._apply_rate_limits(signals)

             # 5. Write signals to DataStore
             written = 0
             for signal in filtered:
                 try:
                     signal_id = self.store.insert_signal(
                         signal_type=signal.signal_type,
                         confidence=signal.confidence,
                         evidence=signal.evidence,
                         suggested_action=signal.suggested_action,
                         ttl_sec=signal.ttl_sec,
                     )
                     self._record_signal(signal.signal_type)
                     written += 1
                     logger.info(
                         "signal: type=%s confidence=%.2f id=%d",
                         signal.signal_type, signal.confidence, signal_id,
                     )
                 except Exception:
                     logger.debug(
                         "signal_engine: failed to write signal", exc_info=True
                     )

             return written
     ```
  2. Export from `src/sigil_ml/signals/__init__.py`:
     ```python
     from sigil_ml.signals.engine import SignalEngine
     ```
- **Files**: `src/sigil_ml/signals/engine.py` (new), `src/sigil_ml/signals/__init__.py` (update)
- **Parallel?**: No -- T031-T035 build on this class.
- **Notes**: The `try/except` around `_process_events_inner()` is critical -- signal processing failures must NEVER crash the poller.
- **Validation**:
  - [ ] `SignalEngine` constructs with all required dependencies
  - [ ] `process_events([])` returns 0 (no events, no signals)
  - [ ] Signal processing failure does not propagate exceptions
  - [ ] Returned count matches number of signals written

### Subtask T031 -- Integrate SignalEngine into EventPoller

- **Purpose**: Hook the SignalEngine into the poller's main loop so signals are processed on every poll cycle.
- **Steps**:
  1. Modify `src/sigil_ml/poller.py`:
     - Add `signal_engine` parameter to `EventPoller.__init__()`:
       ```python
       def __init__(self, store: DataStore, models: dict[str, Any],
                    signal_engine: Any | None = None) -> None:
           self.store = store
           self.stuck = models["stuck"]
           self.activity = models["activity"]
           self.workflow = models["workflow"]
           self.duration = models["duration"]
           self.quality = models["quality"]
           self.signal_engine = signal_engine  # Optional: None when not configured
           # ... rest of existing init ...
       ```
     - Add signal processing at the END of `_poll_once()`, after `self.store.commit()`:
       ```python
       def _poll_once(self) -> None:
           # ... existing polling logic unchanged ...

           self.store.commit()

           # --- Signal detection (additive, does not affect predictions) ---
           if self.signal_engine is not None:
               try:
                   task_id = self.store.get_active_task()
                   task_context = {"task_id": task_id} if task_id else None
                   self.signal_engine.process_events(self._buffer, task_context)
               except Exception:
                   logger.debug("poller: signal engine error (non-fatal)", exc_info=True)
       ```
  2. Signal processing runs AFTER the commit, ensuring:
     - It does not interfere with the prediction write path
     - If it fails, the cursor update and predictions are already committed
     - It has access to the latest buffer including newly classified events
  3. The `signal_engine` parameter is optional (`None` by default) so existing tests and non-signal deployments continue working.
- **Files**: `src/sigil_ml/poller.py`
- **Parallel?**: Depends on T030 (SignalEngine exists).
- **Notes**: This is the most critical integration point. The existing `_poll_once()` logic must remain byte-for-byte identical except for the new block at the end.
- **Validation**:
  - [ ] `EventPoller(store, models)` still works without signal_engine (backward compatible)
  - [ ] `EventPoller(store, models, signal_engine=engine)` processes signals
  - [ ] Signal processing failure does not affect prediction writes
  - [ ] Existing `_predict_and_write()` behavior is unchanged
  - [ ] `store.commit()` is called BEFORE signal processing

### Subtask T032 -- Signal Rate Limiting

- **Purpose**: Prevent signal floods by enforcing per-type and total rate limits.
- **Steps**:
  1. Implement rate limiting methods in `SignalEngine`:
     ```python
     def _apply_rate_limits(self, signals: list[Signal]) -> list[Signal]:
         """Filter signals through rate limiting and cooldown rules."""
         now = time.time()
         self._prune_old_records(now)

         filtered: list[Signal] = []
         for signal in signals:
             # Check dismissed cooldown
             if self._is_type_dismissed(signal.signal_type, now):
                 logger.debug(
                     "signal: suppressed (dismissed) type=%s", signal.signal_type
                 )
                 continue

             # Check per-type rate limit
             if self._is_type_rate_limited(signal.signal_type, now):
                 logger.debug(
                     "signal: rate-limited type=%s", signal.signal_type
                 )
                 continue

             # Check total rate limit
             if self._is_total_rate_limited(now):
                 logger.debug("signal: total rate limit reached")
                 break  # Stop processing remaining signals

             filtered.append(signal)

         return filtered

     def _is_type_rate_limited(self, signal_type: str, now: float) -> bool:
         """Check if this signal type has been emitted within the cooldown window."""
         for st, ts in self._recent_signals:
             if st == signal_type and (now - ts) < SIGNAL_TYPE_COOLDOWN_SEC:
                 return True
         return False

     def _is_total_rate_limited(self, now: float) -> bool:
         """Check if total signal count exceeds the window limit."""
         recent_count = sum(
             1 for _, ts in self._recent_signals
             if (now - ts) < SIGNAL_TOTAL_WINDOW_SEC
         )
         return recent_count >= SIGNAL_TOTAL_MAX

     def _record_signal(self, signal_type: str) -> None:
         """Record that a signal was emitted (for rate limiting)."""
         self._recent_signals.append((signal_type, time.time()))

     def _prune_old_records(self, now: float) -> None:
         """Remove expired rate limiting records."""
         cutoff = now - max(SIGNAL_TYPE_COOLDOWN_SEC, SIGNAL_TOTAL_WINDOW_SEC)
         self._recent_signals = [
             (st, ts) for st, ts in self._recent_signals if ts > cutoff
         ]
     ```
  2. Rate limits per plan.md D7:
     - Max 1 signal per signal_type per 5-minute window
     - Max 10 total signals per 5-minute window
  3. Rate limiting state is in-memory (resets on restart). This is acceptable because:
     - Restart clears the event buffer anyway
     - A brief burst after restart is preferable to missing signals
- **Files**: `src/sigil_ml/signals/engine.py`
- **Parallel?**: Yes -- independent of T033 (cooldown).
- **Validation**:
  - [ ] First signal of a type passes through
  - [ ] Second signal of the same type within 5 minutes is suppressed
  - [ ] Same signal type after 5 minutes passes through
  - [ ] 11th total signal within 5 minutes is suppressed
  - [ ] Old records are pruned to prevent memory growth

### Subtask T033 -- Dismissed Signal Cooldown

- **Purpose**: Suppress a signal type for 30 minutes after the user dismisses a suggestion derived from that signal type.
- **Steps**:
  1. Implement cooldown check and refresh methods in `SignalEngine`:
     ```python
     def _is_type_dismissed(self, signal_type: str, now: float) -> bool:
         """Check if this signal type is in dismissed cooldown."""
         dismissed_at = self._dismissed_types.get(signal_type)
         if dismissed_at is None:
             return False
         if (now - dismissed_at) > DISMISSED_COOLDOWN_SEC:
             del self._dismissed_types[signal_type]
             return False
         return True

     def refresh_dismissed_types(self) -> None:
         """Refresh dismissed signal types from feedback data.

         Reads recent feedback from the DataStore and updates the
         cooldown state. Called periodically (e.g., every prediction cycle).
         """
         now = time.time()
         now_ms = int(now * 1000)
         since_ms = now_ms - (DISMISSED_COOLDOWN_SEC * 1000)

         try:
             feedback = self.store.get_signal_feedback(since_ms)
             for fb in feedback:
                 if fb["status"] == "dismissed":
                     signal_type = fb["signal_type"]
                     dismissed_at = fb["created_at"] / 1000.0
                     self._dismissed_types[signal_type] = dismissed_at
         except Exception:
             logger.debug("signal_engine: failed to refresh dismissed types", exc_info=True)
     ```
  2. `refresh_dismissed_types()` should be called periodically, not on every poll cycle (too expensive). Call it once per prediction cycle (every 60s) or when `_predict_and_write()` runs.
  3. The cooldown is 30 minutes per spec D7. After 30 minutes, the signal type is no longer suppressed.
  4. If `get_signal_feedback()` fails (e.g., suggestions table doesn't have signal_id column yet), fail silently.
- **Files**: `src/sigil_ml/signals/engine.py`
- **Parallel?**: Yes -- independent of T032 (rate limiting).
- **Validation**:
  - [ ] After dismissal, the signal type is suppressed for 30 minutes
  - [ ] After 30 minutes, the signal type is no longer suppressed
  - [ ] Feedback refresh failure does not crash the engine
  - [ ] Dismissed types are cleaned up after cooldown expires

### Subtask T034 -- Wire SignalEngine in app.py

- **Purpose**: Initialize the SignalEngine and all signal model instances during local-mode startup, and pass the engine to the EventPoller.
- **Steps**:
  1. Modify `src/sigil_ml/app.py` -- in the local-mode startup block of `lifespan()`:
     ```python
     # After existing model loading...
     state.load_models(ms)

     # Initialize signal pipeline (additive, does not modify existing models)
     from sigil_ml.signals.profile import BehaviorProfile
     from sigil_ml.signals.pattern_detector import PatternDetector
     from sigil_ml.signals.next_action import NextActionPredictor
     from sigil_ml.signals.file_recommender import FileRecommender
     from sigil_ml.signals.engine import SignalEngine

     profile = BehaviorProfile()
     pattern_detector = PatternDetector()
     next_action_predictor = NextActionPredictor()
     file_recommender = FileRecommender()

     # Load persisted signal models
     next_action_predictor.load(ms)
     file_recommender.load(ms)
     pattern_detector.load(ms)

     signal_engine = SignalEngine(
         store=store,
         profile=profile,
         pattern_detector=pattern_detector,
         next_action=next_action_predictor,
         file_recommender=file_recommender,
     )
     state.signal_engine = signal_engine

     state.poller = EventPoller(
         store=store,
         models={
             "stuck": state.stuck,
             "activity": state.activity,
             "workflow": state.workflow,
             "duration": state.duration,
             "quality": state.quality,
         },
         signal_engine=signal_engine,
     )
     ```
  2. Add `signal_engine` field to `AppState`:
     ```python
     class AppState:
         def __init__(self, mode: ServingMode = ServingMode.LOCAL) -> None:
             # ... existing fields ...
             self.signal_engine: Any = None
     ```
  3. Cloud mode does NOT initialize the SignalEngine -- signals are a local-only feature in this phase.
  4. The `reload_models_into_poller()` method should also reload signal models after retraining:
     ```python
     def reload_models_into_poller(self) -> None:
         self.load_models()
         if self.poller:
             self.poller.stuck = self.stuck
             # ... existing reloads ...
         if self.signal_engine and self.model_store:
             self.signal_engine.pattern_detector.load(self.model_store)
             self.signal_engine.next_action.load(self.model_store)
             self.signal_engine.file_recommender.load(self.model_store)
         logger.info("models reloaded into poller")
     ```
- **Files**: `src/sigil_ml/app.py`
- **Parallel?**: Depends on T030 (SignalEngine class exists).
- **Notes**: Imports are inside the lifespan function to keep module-level imports unchanged.
- **Validation**:
  - [ ] Local mode startup creates SignalEngine and passes to poller
  - [ ] Cloud mode startup does NOT create SignalEngine
  - [ ] `state.signal_engine` is accessible from routes if needed
  - [ ] Model reloading refreshes signal models
  - [ ] Existing startup behavior is unchanged (same logs, same model loading)

### Subtask T035 -- Write Profile to ml_predictions

- **Purpose**: Persist the BehaviorProfile to `ml_predictions` on each prediction cycle so the Go daemon can read it for LLM context.
- **Steps**:
  1. Add profile writing to the poller's prediction cycle. In `_predict_and_write()` at the end (before audit log):
     ```python
     def _predict_and_write(self) -> None:
         # ... existing prediction logic unchanged ...

         # Write behavior profile (signal pipeline)
         if self.signal_engine is not None:
             try:
                 profile_data = self.signal_engine.profile.to_dict()
                 self.store.insert_prediction("profile", profile_data, 1.0, None)
                 # Refresh dismissed signal types while we're here
                 self.signal_engine.refresh_dismissed_types()
             except Exception:
                 logger.debug("poller: profile write failed (non-fatal)", exc_info=True)

         # Audit log
         latency_ms = int((time.time() - start) * 1000)
         self.store.insert_ml_event("prediction", "poller", "local", latency_ms)
     ```
  2. Profile is written with `model='profile'`, no TTL (overwritten each cycle).
  3. This also calls `refresh_dismissed_types()` once per prediction cycle (every 60s), which is a good cadence for feedback refresh.
  4. Profile write failure is non-fatal -- the profile is still in memory.
- **Files**: `src/sigil_ml/poller.py`
- **Parallel?**: Depends on T031 (signal_engine is on the poller).
- **Notes**: The Go daemon will read the profile from `ml_predictions` where `model = 'profile'` to include user context in LLM prompts.
- **Validation**:
  - [ ] `ml_predictions` table gains rows with `model='profile'` after prediction cycles
  - [ ] Profile data is valid JSON matching data-model.md BehaviorProfile structure
  - [ ] Profile write failure does not affect other predictions
  - [ ] `refresh_dismissed_types()` is called on each prediction cycle

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Signal processing slows down poll cycle | Medium | High | Error-isolated try/except; runs AFTER commit; target <500ms |
| Breaking existing prediction pipeline | Medium | Critical | Signal_engine is optional parameter; all changes are additive |
| Rate limiting state lost on restart | Expected | Low | Brief signal burst after restart is acceptable; in-memory state sufficient |
| Feedback read failure blocks signal cooldown | Low | Low | Graceful fallback: no cooldown if feedback unavailable |

## Review Guidance

- **Zero regression**: This is the most critical review point. Verify that `_poll_once()` and `_predict_and_write()` produce IDENTICAL behavior when `signal_engine is None`. Compare the modified code line-by-line against the current version.
- **Error isolation**: Verify that every signal-related code path is wrapped in try/except with debug-level logging.
- **Rate limiting correctness**: Verify the 5-minute per-type and 10-total limits work correctly with the pruning logic.
- **Poller constructor**: Verify `signal_engine=None` default preserves backward compatibility for existing tests.
- **App startup order**: Verify signal models are loaded AFTER the DataStore and ModelStore are initialized.
- **Profile write**: Verify profile is written to `ml_predictions` with `model='profile'` and no TTL.

---

## Activity Log

- 2026-03-30T18:27:35Z -- system -- lane=planned -- Prompt generated via /spec-kitty.tasks.
