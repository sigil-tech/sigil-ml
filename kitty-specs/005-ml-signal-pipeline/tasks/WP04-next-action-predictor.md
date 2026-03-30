---
work_package_id: "WP04"
title: "Next-Action Predictor"
lane: "planned"
dependencies: ["WP02"]
subtasks:
  - "T019"
  - "T020"
  - "T021"
  - "T022"
  - "T023"
  - "T024"
phase: "Phase 1 - Signal Models"
assignee: ""
agent: ""
shell_pid: ""
review_status: ""
reviewed_by: ""
requirement_refs:
  - "FR-005"
  - "FR-010"
history:
  - timestamp: "2026-03-30T18:27:35Z"
    lane: "planned"
    agent: "system"
    shell_pid: ""
    action: "Prompt generated via /spec-kitty.tasks"
---

# Work Package Prompt: WP04 -- Next-Action Predictor

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
spec-kitty implement WP04 --base WP02
```

Depends on WP02 (BehaviorProfile for profile-based filtering). Can run in parallel with WP03 and WP05.

---

## Objectives & Success Criteria

1. `NextActionPredictor` maintains n-gram frequency tables trained from event sequences.
2. It predicts the most likely next action given the recent event context.
3. When the actual next action has < 5% predicted probability, a divergence signal is emitted.
4. No signals are emitted during cold start (< 1000 events processed).
5. Composite action tokens (`category:tool`) capture tool-specific patterns.
6. N-gram tables persist via ModelStore and survive restarts.

## Context & Constraints

- **Spec**: FR-005 (next-action prediction, divergence signals), FR-010 (cold-start, silence over noise).
- **Plan**: Design D4 (n-gram on composite tokens), D6 (composite action token extraction).
- **Research**: R4 confirms n-gram approach. Order 3 with backoff to 2 and 1.
- **Data Model**: CompositeActionToken format: `{category}:{tool}` (e.g., `verifying:pytest`, `editing:python`).
- **Event Classification**: Events arrive with `_category` key from `ActivityClassifier.classify()`. Categories: editing, verifying, navigating, researching, integrating, communicating, idle.
- **Existing Features**: `src/sigil_ml/features.py` contains event kind detection patterns that will be reused for tool inference.

---

## Subtasks & Detailed Guidance

### Subtask T019 -- Create NextActionPredictor Class

- **Purpose**: Define the `NextActionPredictor` class skeleton with the `check_divergence()` entry point.
- **Steps**:
  1. Create `src/sigil_ml/signals/next_action.py`:
     ```python
     """Next-action prediction using n-gram models on composite action tokens.

     Learns typical action sequences from event history. Emits divergence
     signals when the user's actual behavior has low predicted probability.
     """

     from __future__ import annotations

     import logging
     from collections import Counter, defaultdict
     from typing import Any

     from sigil_ml.signals import Signal
     from sigil_ml.signals.profile import BehaviorProfile

     logger = logging.getLogger(__name__)

     # Minimum events before the predictor starts emitting signals.
     MIN_EVENTS_FOR_PREDICTION = 1000
     # Divergence threshold: signal when actual probability < this.
     DIVERGENCE_THRESHOLD = 0.05
     # Minimum total n-gram count for a context to be predictive.
     MIN_CONTEXT_COUNT = 10


     class NextActionPredictor:
         """Predicts likely next actions from recent event sequences.

         Uses n-gram frequency tables (order 3 with backoff to 2 and 1)
         on composite action tokens ({category}:{tool}).
         """

         def __init__(self, n: int = 3) -> None:
             self._n = n
             self._ngrams: dict[tuple, Counter] = defaultdict(Counter)
             self._total_tokens: int = 0

         def check_divergence(
             self, buffer: list[dict], profile: BehaviorProfile
         ) -> list[Signal]:
             """Check for divergence between predicted and actual behavior.

             Args:
                 buffer: Recent classified events from the poller.
                 profile: Current user behavior profile.

             Returns:
                 List of Signal objects for detected divergences.
             """
             if self._total_tokens < MIN_EVENTS_FOR_PREDICTION:
                 return []

             tokens = self._extract_tokens(buffer)
             if len(tokens) < self._n:
                 return []

             return self._check_latest_divergence(tokens, profile)
     ```
  2. Export from `src/sigil_ml/signals/__init__.py`:
     ```python
     from sigil_ml.signals.next_action import NextActionPredictor
     ```
- **Files**: `src/sigil_ml/signals/next_action.py` (new), `src/sigil_ml/signals/__init__.py` (update)
- **Parallel?**: No -- T021-T024 build on this skeleton.
- **Validation**:
  - [ ] `NextActionPredictor()` constructs without errors
  - [ ] `check_divergence([], profile)` returns empty list
  - [ ] Returns empty list when total_tokens < 1000 (cold start)

### Subtask T020 -- Composite Action Token Extraction

- **Purpose**: Implement `extract_action_token()` and `infer_tool()` helper functions that convert events into composite tokens.
- **Steps**:
  1. Add to `src/sigil_ml/features.py` (at the end, after existing functions):
     ```python
     # --- Composite action token extraction (for signal pipeline) ---

     def extract_action_token(event: dict) -> str:
         """Convert a classified event into a composite action token.

         Format: "{category}:{tool}" when tool is identifiable,
                 "{category}" when tool is unknown.

         Examples: "verifying:pytest", "editing:py", "integrating:git"
         """
         category = event.get("_category", "idle")
         tool = infer_tool(event)
         return f"{category}:{tool}" if tool else category


     def infer_tool(event: dict) -> str | None:
         """Infer the specific tool from an event's payload.

         Returns:
             Tool identifier string, or None if cannot be determined.
         """
         kind = event.get("kind", "")
         payload = event.get("payload") or {}
         if isinstance(payload, str):
             return None

         if kind == "terminal":
             cmd = str(payload.get("cmd", "")).strip().split()
             if cmd:
                 return cmd[0].split("/")[-1]
             return None

         if kind == "process":
             comm = str(payload.get("comm", "")).split("/")[-1].strip("()")
             return comm if comm else None

         if kind == "git":
             return "git"

         if kind == "file":
             path = str(payload.get("path", ""))
             if "." in path:
                 return path.rsplit(".", 1)[-1].lower()
             return "unknown"

         if kind == "ai":
             source = event.get("source", "")
             return source if source else "ai"

         return None
     ```
  2. These functions are in `features.py` (not in the signals package) because they are general-purpose feature extraction utilities that may be used by other modules.
  3. The logic matches plan.md D6 exactly. Tool inference patterns:
     - Terminal: first word of command (e.g., `pytest`, `git`)
     - Process: command name from `comm` field
     - Git: always `"git"`
     - File: file extension (e.g., `py`, `go`, `md`)
     - AI: source field
- **Files**: `src/sigil_ml/features.py` (add ~40 lines at end)
- **Parallel?**: Yes -- modifies a different file from T019/T021.
- **Notes**: `extract_action_token()` depends on `_category` being set on the event (done by ActivityClassifier in poller.py).
- **Validation**:
  - [ ] `extract_action_token({"kind": "terminal", "payload": {"cmd": "pytest tests/"}, "_category": "verifying"})` returns `"verifying:pytest"`
  - [ ] `extract_action_token({"kind": "file", "payload": {"path": "src/main.py"}, "_category": "editing"})` returns `"editing:py"`
  - [ ] `extract_action_token({"kind": "git", "_category": "integrating"})` returns `"integrating:git"`
  - [ ] `extract_action_token({"_category": "idle"})` returns `"idle"` (no tool)

### Subtask T021 -- N-Gram Frequency Table

- **Purpose**: Implement the n-gram data structure with prediction and backoff.
- **Steps**:
  1. Add methods to `NextActionPredictor`:
     ```python
     def predict(self, recent_tokens: list[str]) -> dict[str, float] | None:
         """Predict probability distribution over next token.

         Uses n-gram with backoff: tries order n, then n-1, then 1.

         Args:
             recent_tokens: Recent composite action tokens.

         Returns:
             Dict mapping token -> probability for top predictions,
             or None if no prediction is possible.
         """
         # Try decreasing context lengths
         for order in range(self._n, 0, -1):
             if len(recent_tokens) < order - 1:
                 continue
             context = tuple(recent_tokens[-(order - 1):]) if order > 1 else ()
             counts = self._ngrams.get(context)
             if counts is None:
                 continue
             total = sum(counts.values())
             if total < MIN_CONTEXT_COUNT:
                 continue
             return {
                 token: count / total
                 for token, count in counts.most_common(10)
             }
         return None

     def _extract_tokens(self, buffer: list[dict]) -> list[str]:
         """Convert an event buffer into a list of composite action tokens."""
         from sigil_ml.features import extract_action_token
         return [extract_action_token(e) for e in buffer]
     ```
  2. The backoff strategy ensures predictions even with sparse n-gram data:
     - First try trigram context (t-2, t-1)
     - If insufficient data, try bigram (t-1)
     - If still insufficient, try unigram (no context)
  3. `MIN_CONTEXT_COUNT = 10` prevents predictions from very sparse contexts.
- **Files**: `src/sigil_ml/signals/next_action.py`
- **Parallel?**: Depends on T019 skeleton.
- **Validation**:
  - [ ] After training with ["A", "A", "B", "A", "A", "B"] * 100, predict(["A", "A"]) returns {"B": ~1.0}
  - [ ] Backoff works: if trigram context is unseen, bigram context is tried
  - [ ] Returns None when no context has sufficient counts

### Subtask T022 -- Incremental N-Gram Training

- **Purpose**: Build n-gram frequency tables incrementally from event sequences.
- **Steps**:
  1. Implement `train_incremental()`:
     ```python
     def train_incremental(self, tokens: list[str]) -> None:
         """Update n-gram tables from a token sequence.

         Called on every poll cycle with the full buffer's tokens.
         For full rebuild (training), call with all historical tokens.

         Args:
             tokens: List of composite action tokens.
         """
         if len(tokens) < self._n:
             return

         # Build n-grams for all orders (1 through n)
         for order in range(1, self._n + 1):
             for i in range(len(tokens) - order + 1):
                 if order > 1:
                     context = tuple(tokens[i:i + order - 1])
                     next_token = tokens[i + order - 1]
                 else:
                     context = ()
                     next_token = tokens[i]
                 self._ngrams[context][next_token] += 1

         self._total_tokens += len(tokens)
     ```
  2. Training is incremental -- each poll cycle adds the buffer's tokens to the n-gram tables.
  3. For full retraining (WP07), the trainer will call `train_incremental()` with complete event sequences from all completed tasks.
  4. Add a `reset()` method for full retraining:
     ```python
     def reset(self) -> None:
         """Clear all n-gram tables for full retraining."""
         self._ngrams = defaultdict(Counter)
         self._total_tokens = 0
     ```
- **Files**: `src/sigil_ml/signals/next_action.py`
- **Parallel?**: Depends on T019.
- **Notes**: Incremental training means n-gram counts grow monotonically. For the poller use case, this is fine -- the buffer represents new events not yet seen. For full retraining, call `reset()` first.
- **Validation**:
  - [ ] After training on a known sequence, n-gram counts match expected values
  - [ ] `total_tokens` correctly tracks cumulative token count
  - [ ] `reset()` clears all tables and resets counter to 0
  - [ ] Works correctly with sequences shorter than n (no errors, no updates)

### Subtask T023 -- Divergence Detection

- **Purpose**: Detect when the user's actual behavior diverges from predicted next action.
- **Steps**:
  1. Implement `_check_latest_divergence()`:
     ```python
     def _check_latest_divergence(
         self, tokens: list[str], profile: BehaviorProfile
     ) -> list[Signal]:
         """Check if the most recent action diverges from prediction."""
         signals: list[Signal] = []

         if len(tokens) < self._n:
             return signals

         # Get prediction for what the next token should be
         context_tokens = tokens[:-1]
         actual_token = tokens[-1]
         prediction = self.predict(context_tokens)

         if prediction is None:
             return signals  # No confident prediction possible

         # Check if actual action has low predicted probability
         actual_prob = prediction.get(actual_token, 0.0)
         top_predicted = max(prediction, key=prediction.get)
         top_prob = prediction[top_predicted]

         if actual_prob < DIVERGENCE_THRESHOLD and top_prob > 0.3:
             # Divergence: actual action was unexpected AND we had a confident prediction
             confidence = min((top_prob - actual_prob) / top_prob, 0.95)
             signals.append(Signal(
                 signal_type="action_divergence",
                 confidence=round(confidence, 4),
                 evidence={
                     "source_model": "next_action",
                     "predicted_action": top_predicted,
                     "predicted_probability": round(top_prob, 4),
                     "actual_action": actual_token,
                     "actual_probability": round(actual_prob, 4),
                     "sequence_length": len(tokens),
                     "context": {
                         "recent_tokens": tokens[-5:],
                     },
                 },
                 suggested_action=self._action_hint(top_predicted),
             ))

         return signals

     def _action_hint(self, predicted_token: str) -> str | None:
         """Infer a suggested action from the predicted token."""
         category = predicted_token.split(":")[0] if ":" in predicted_token else predicted_token
         hint_map = {
             "verifying": "test",
             "integrating": "commit",
             "navigating": None,
             "researching": None,
             "editing": None,
             "idle": "take_break",
         }
         return hint_map.get(category)
     ```
  2. Divergence requires TWO conditions:
     - Actual action has < 5% predicted probability (unexpected)
     - Top predicted action has > 30% probability (model is confident)
  3. This prevents noisy signals when the model is uncertain about everything.
- **Files**: `src/sigil_ml/signals/next_action.py`
- **Parallel?**: Depends on T021 (n-gram table).
- **Notes**: Evidence JSON matches data-model.md next-action divergence format.
- **Validation**:
  - [ ] Divergence signal emitted when actual has < 5% prob and top prediction > 30%
  - [ ] No signal when actual matches prediction (high probability)
  - [ ] No signal when model is uncertain (top prediction < 30%)
  - [ ] Evidence contains predicted_action, actual_action, probabilities, recent_tokens

### Subtask T024 -- Model Persistence via ModelStore

- **Purpose**: Save and load n-gram tables so the model survives process restarts.
- **Steps**:
  1. Implement save/load methods:
     ```python
     def save(self, model_store) -> None:
         """Persist n-gram tables via ModelStore."""
         import io
         import joblib
         data = {
             "ngrams": dict(self._ngrams),  # Convert defaultdict to regular dict
             "total_tokens": self._total_tokens,
             "n": self._n,
         }
         buf = io.BytesIO()
         joblib.dump(data, buf)
         model_store.save("next_action", buf.getvalue())
         logger.info("NextActionPredictor: saved %d n-gram contexts", len(self._ngrams))

     def load(self, model_store) -> bool:
         """Load n-gram tables from ModelStore. Returns True if loaded."""
         raw = model_store.load("next_action")
         if raw is None:
             return False
         import io
         import joblib
         try:
             data = joblib.load(io.BytesIO(raw))
             loaded_ngrams = data.get("ngrams", {})
             self._ngrams = defaultdict(Counter)
             for context, counts in loaded_ngrams.items():
                 self._ngrams[context] = Counter(counts)
             self._total_tokens = data.get("total_tokens", 0)
             self._n = data.get("n", self._n)
             logger.info(
                 "NextActionPredictor: loaded %d contexts, %d total tokens",
                 len(self._ngrams), self._total_tokens,
             )
             return True
         except Exception:
             logger.warning("NextActionPredictor: failed to load model", exc_info=True)
             return False
     ```
  2. Follow the `ActivityClassifier` pattern: `save()` via `model_store.save(name, bytes)`, `load()` via `model_store.load(name) -> bytes | None`.
  3. The model name is `"next_action"` -- consistent with `model_store.save("next_action", ...)`.
  4. On startup (in WP06 app.py wiring), `load()` is called to restore state.
- **Files**: `src/sigil_ml/signals/next_action.py`
- **Parallel?**: Depends on T019.
- **Notes**: `joblib` is used for serialization (already a project dependency). The n-gram dict with Counter values serializes cleanly.
- **Validation**:
  - [ ] `save()` followed by `load()` restores identical n-gram tables
  - [ ] `total_tokens` is preserved across save/load
  - [ ] `load()` returns False when no model exists
  - [ ] `load()` returns False and logs warning on corrupted data

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| N-gram vocabulary grows unboundedly | Medium | Medium | Bounded by observed behavior; periodic pruning in training (WP07) |
| Divergence signals too frequent | Medium | Medium | Dual threshold (< 5% actual AND > 30% predicted) reduces noise; rate limiting in WP06 |
| Token extraction mismatch between features.py and signals | Low | High | Single implementation in features.py used by both profile and predictor |
| Incremental training drift | Low | Medium | Full retraining in WP07 rebuilds from scratch periodically |

## Review Guidance

- **Token format**: Verify `extract_action_token()` output matches data-model.md CompositeActionToken format exactly.
- **Backoff correctness**: Verify the n-gram backoff tries order n, then n-1, ..., then 1 (not just n and 1).
- **Cold-start safety**: Verify no signals when `total_tokens < 1000`.
- **Dual threshold**: Verify divergence requires BOTH low actual probability AND high predicted probability.
- **Evidence structure**: Verify evidence JSON matches data-model.md next-action divergence format.
- **features.py non-regression**: Verify existing functions (`extract_activity_features`, `extract_stuck_features`, etc.) are unchanged.

---

## Activity Log

- 2026-03-30T18:27:35Z -- system -- lane=planned -- Prompt generated via /spec-kitty.tasks.
