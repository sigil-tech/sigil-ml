---
work_package_id: "WP04"
title: "Refactor All Model Classes to Use ModelStore"
lane: "planned"
dependencies: ["WP01"]
subtasks:
  - "T017"
  - "T018"
  - "T019"
  - "T020"
  - "T021"
  - "T022"
phase: "Phase 2 - Model Refactor"
assignee: ""
agent: ""
shell_pid: ""
review_status: ""
reviewed_by: ""
history:
  - timestamp: "2026-03-30T01:45:11Z"
    lane: "planned"
    agent: "system"
    shell_pid: ""
    action: "Prompt generated via /spec-kitty.tasks"
requirement_refs:
  - "FR-007"
  - "FR-011"
---

# Work Package Prompt: WP04 -- Refactor All Model Classes to Use ModelStore

## Review Feedback

> **Populated by `/spec-kitty.review`** -- Reviewers add detailed feedback here when work needs changes.

*[This section is empty initially.]*

---

## Markdown Formatting
Wrap HTML/XML tags in backticks: `` `<div>` ``, `` `<script>` ``
Use language identifiers in code blocks: ````python`, ````bash`

---

## Objectives & Success Criteria

- Refactor all 5 model classes to accept a `ModelStore` instance via constructor injection.
- Remove all direct filesystem I/O for weight persistence from model classes.
- Preserve identical prediction behavior -- only the load/save path changes.
- Ensure fallback to rule-based predictions when `ModelStore.load()` returns `None`.

**Success Criteria (SC-002 from spec)**:
- No model class directly calls `config.weights_path()`, `joblib.load(path)`, `joblib.dump(obj, path)`, `open(path)`, or uses `pathlib.Path` for weight file I/O.
- Each model's `__init__` accepts `model_store: ModelStore | None = None`. When `None`, creates a `LocalModelStore()` internally.
- `load()` + `BytesIO` deserialization replaces direct file reads for joblib-based models.
- `save()` + `BytesIO` serialization replaces direct file writes for joblib-based models.
- `QualityEstimator` uses `json.dumps/loads` with `encode/decode("utf-8")` instead of `open()`.
- All prediction logic is completely untouched.
- Rule-based fallback behavior is preserved exactly.

## Context & Constraints

- **Spec**: `kitty-specs/003-model-storage-abstraction/spec.md` -- FR-007, FR-011.
- **Plan**: Design decision D2 (serialization responsibility -- models own serialization, store handles bytes), D8 (refactoring approach -- optional `model_store` param with backward-compatible default).
- **Depends on WP01**: `ModelStore` protocol and `LocalModelStore` in `src/sigil_ml/storage/model_store.py`.
- **Current model files** (all in `src/sigil_ml/models/`):
  - `stuck.py`: `StuckPredictor` -- uses `config.weights_path("stuck")` + `joblib.load(path)` / `joblib.dump(model, path)`. Fallback: returns `{"probability": 0.5, "confidence": "weak"}`.
  - `activity.py`: `ActivityClassifier` -- uses `config.weights_path("activity")` + `joblib.load(path)` / `joblib.dump(model, path)`. Fallback: `_classify_rules()`.
  - `workflow.py`: `WorkflowStatePredictor` -- uses `config.weights_path("workflow")` + `joblib.load(path)` / `joblib.dump(model, path)`. Fallback: `_predict_rules()`.
  - `duration.py`: `DurationEstimator` -- uses `config.weights_path("duration")` + `joblib.load(path)` / `joblib.dump(model, path)`. Fallback: returns `{"estimated_minutes": 60.0, "confidence_interval": [30.0, 90.0]}`.
  - `quality.py`: `QualityEstimator` -- uses `config.weights_path("quality")` + `json.load(open(path))` / `json.dump(data, open(path))`. Uses `pathlib.Path`. Fallback: uses `DEFAULT_WEIGHTS`.
- **Constraint**: `joblib` import stays in model classes but is used via `BytesIO` instead of file paths.
- **Constraint**: `config` import can remain for non-persistence concerns, but `config.weights_path()` must not be called.
- **Constraint**: Use `TYPE_CHECKING` guard for `ModelStore` import to avoid circular imports.

## Subtasks & Detailed Guidance

### Subtask T017 -- Refactor StuckPredictor to use ModelStore

- **Purpose**: Convert the first model class as a reference pattern for the remaining four. `StuckPredictor` is the simplest model and a clean example.

- **Steps**:
  1. Modify `src/sigil_ml/models/stuck.py`:
     ```python
     """Stuck predictor using GradientBoostingClassifier."""

     from __future__ import annotations

     import logging
     from io import BytesIO
     from typing import TYPE_CHECKING

     import joblib
     import numpy as np
     from sklearn.ensemble import GradientBoostingClassifier

     if TYPE_CHECKING:
         from sigil_ml.storage import ModelStore

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

         MODEL_NAME = "stuck"

         def __init__(self, model_store: "ModelStore | None" = None) -> None:
             if model_store is None:
                 from sigil_ml.storage import LocalModelStore
                 model_store = LocalModelStore()
             self._store = model_store
             self.model: GradientBoostingClassifier | None = None
             self._trained = False
             self._load()

         def _load(self) -> None:
             data = self._store.load(self.MODEL_NAME)
             if data is not None:
                 try:
                     self.model = joblib.load(BytesIO(data))
                     self._trained = True
                     logger.info("Loaded stuck model from store")
                 except Exception:
                     logger.warning("Failed to deserialize stuck model, starting fresh")
                     if hasattr(self._store, "evict"):
                         self._store.evict(self.MODEL_NAME)
                     self.model = None

         @property
         def is_trained(self) -> bool:
             return self._trained

         def predict(self, features: dict) -> dict:
             # ... COMPLETELY UNCHANGED from current code ...

         def train(self, X: np.ndarray, y: np.ndarray) -> None:
             self.model = GradientBoostingClassifier(
                 n_estimators=100,
                 max_depth=3,
                 learning_rate=0.1,
                 random_state=42,
             )
             self.model.fit(X, y)
             self._trained = True

             buf = BytesIO()
             joblib.dump(self.model, buf)
             self._store.save(self.MODEL_NAME, buf.getvalue())
             logger.info("Saved stuck model to store")
     ```
  2. **Remove**: `from sigil_ml import config` (no longer needed in this file).
  3. **Add**: `from __future__ import annotations`, `from io import BytesIO`, `TYPE_CHECKING` block.
  4. **Do NOT touch**: The `predict()` method body, `FEATURE_NAMES`, `is_trained` property.

- **Files**:
  - Update: `src/sigil_ml/models/stuck.py`

- **Parallel?**: No -- this is the reference pattern. Complete T017 first, then T018-T021 in parallel.

- **Notes**:
  - `BytesIO(data)` wraps raw bytes so `joblib.load()` can read it as a file-like object.
  - `BytesIO()` for writing: `joblib.dump(model, buf)` writes to the in-memory buffer; `buf.getvalue()` extracts bytes.
  - The `MODEL_NAME = "stuck"` class constant replaces hardcoded string literals.
  - The `evict` call on deserialization failure is defensive: if the store is a `CachedModelStore` (WP03), it evicts the corrupted cache entry. For `LocalModelStore`, `hasattr(store, 'evict')` returns `False`.
  - The default `model_store=None` means `StuckPredictor()` (no args) still works -- backward compatible.

### Subtask T018 -- Refactor ActivityClassifier to use ModelStore

- **Purpose**: Apply the same pattern from T017 to the activity classifier.

- **Steps**:
  1. Modify `src/sigil_ml/models/activity.py`:
     - Add `model_store: "ModelStore | None" = None` to `__init__`.
     - Create `_load()` method with store-based loading (same pattern as T017).
     - Update `_save()` to serialize via `BytesIO` + `joblib.dump` and call `self._store.save()`.
     - Remove `from sigil_ml import config` if only used for `weights_path`.
     - Add `from __future__ import annotations`, `from io import BytesIO`, `TYPE_CHECKING` guard.
  2. **Do NOT touch**: `classify()`, `classify_batch()`, `_classify_rules()`, `_classify_ml()`, `train()` logic (only the `_save()` call within `train()` changes), category constants, verify/integrate prefixes.

- **Files**:
  - Update: `src/sigil_ml/models/activity.py`

- **Parallel?**: Yes -- independent file from T017, T019, T020, T021.

- **Notes**:
  - `ActivityClassifier` has a `_save()` method that `train()` calls. Only `_save()` needs updating -- `train()` logic is unchanged.
  - Current `__init__` loads weights in the body. Refactor to a `_load()` method for consistency with T017 pattern.
  - The `features.py` import (`extract_activity_features`) remains -- it's for prediction, not persistence.

### Subtask T019 -- Refactor WorkflowStatePredictor to use ModelStore

- **Purpose**: Apply the same pattern to the workflow state predictor.

- **Steps**:
  1. Modify `src/sigil_ml/models/workflow.py`:
     - Add `model_store: "ModelStore | None" = None` to `__init__`.
     - Create `_load()` method with store-based loading.
     - Update `_save()` to use `BytesIO` + `joblib.dump` + `store.save()`.
     - Remove `from sigil_ml import config` if only used for `weights_path`.
     - Add imports: `from __future__ import annotations`, `from io import BytesIO`, `TYPE_CHECKING`.
  2. **Do NOT touch**: `predict()`, `_predict_rules()`, `_predict_ml()`, `_activity_distribution()`, `_compute_momentum()`, `_compute_focus()`, `train()` logic (only `_save()` call changes).

- **Files**:
  - Update: `src/sigil_ml/models/workflow.py`

- **Parallel?**: Yes -- independent file.

- **Notes**:
  - `WorkflowStatePredictor` has many static helper methods that are purely computational. These are 100% unrelated to persistence and must not be modified.
  - The `features.py` import (`extract_workflow_features`) remains.

### Subtask T020 -- Refactor DurationEstimator to use ModelStore

- **Purpose**: Apply the same pattern to the duration estimator.

- **Steps**:
  1. Modify `src/sigil_ml/models/duration.py`:
     - Add `model_store: "ModelStore | None" = None` to `__init__`.
     - Create `_load()` method with store-based loading.
     - Update `train()` to serialize via `BytesIO` and call `store.save()`.
     - Remove `from sigil_ml import config`.
     - Add imports: `from __future__ import annotations`, `from io import BytesIO`, `TYPE_CHECKING`.
  2. **Do NOT touch**: `predict()` method (includes the `estimators_` confidence interval logic), `FEATURE_NAMES`, `is_trained` property.

- **Files**:
  - Update: `src/sigil_ml/models/duration.py`

- **Parallel?**: Yes -- independent file.

- **Notes**:
  - `DurationEstimator.predict()` accesses `self.model.estimators_.flatten()` for confidence intervals. This internal sklearn attribute is preserved correctly by `joblib` byte-round-trip serialization.
  - Unlike `ActivityClassifier` and `WorkflowStatePredictor`, `DurationEstimator` does not have a separate `_save()` method -- the save logic is inline in `train()`. Refactor to either add a `_save()` method or keep inline.

### Subtask T021 -- Refactor QualityEstimator to use ModelStore (JSON serialization)

- **Purpose**: Apply the store pattern to QualityEstimator. **This model is different** -- it uses JSON serialization, not joblib. The weights are a simple `{str: int}` dict, not a sklearn model object.

- **Steps**:
  1. Modify `src/sigil_ml/models/quality.py`:
     ```python
     """Work quality estimator -- rolling 30-minute quality score."""

     from __future__ import annotations

     import json
     import logging
     from typing import TYPE_CHECKING

     import numpy as np

     if TYPE_CHECKING:
         from sigil_ml.storage import ModelStore

     logger = logging.getLogger(__name__)

     # ... DEFAULT_WEIGHTS, THRESHOLD_LOW, THRESHOLD_HIGH unchanged ...


     class QualityEstimator:
         MODEL_NAME = "quality"

         def __init__(self, model_store: "ModelStore | None" = None) -> None:
             if model_store is None:
                 from sigil_ml.storage import LocalModelStore
                 model_store = LocalModelStore()
             self._store = model_store
             self.weights = dict(DEFAULT_WEIGHTS)
             self._load_weights()

         def _load_weights(self) -> None:
             data = self._store.load(self.MODEL_NAME)
             if data is not None:
                 try:
                     saved = json.loads(data.decode("utf-8"))
                     self.weights = saved.get("weights", self.weights)
                     logger.info("quality: loaded learned weights from store")
                 except Exception as e:
                     logger.warning("quality: failed to load weights: %s", e)
                     if hasattr(self._store, "evict"):
                         self._store.evict(self.MODEL_NAME)

         def _save_weights(self) -> None:
             data = json.dumps({"weights": self.weights}).encode("utf-8")
             self._store.save(self.MODEL_NAME, data)

         # predict() and train() UNCHANGED except train() calls self._save_weights()
     ```
  2. **Remove**: `from pathlib import Path` import (no longer needed).
  3. **Remove**: `from sigil_ml import config` import (no longer needed for weights).
  4. **Keep**: `json` import (used for serialization, NOT file I/O).
  5. **Do NOT touch**: `predict()` method, `train()` logic (only `_save_weights()` call within `train()` is already there), constants.

- **Files**:
  - Update: `src/sigil_ml/models/quality.py`

- **Parallel?**: Yes -- independent file.

- **Notes**:
  - **Key difference from other models**: Uses `json.dumps({"weights": self.weights}).encode("utf-8")` for save and `json.loads(data.decode("utf-8"))` for load. No `joblib` or `BytesIO` needed.
  - The current code saves as JSON with a `.joblib` file extension (via `config.weights_path("quality")`). The `LocalModelStore` preserves this -- all files use `.joblib` extension regardless of content. This is cosmetic and harmless.
  - The current `_load_weights()` uses `Path(config.weights_path("quality"))` then `open(path)` then `json.load(f)`. Replace with `self._store.load(self.MODEL_NAME)` then `json.loads(data.decode("utf-8"))`.
  - The current `_save_weights()` uses `path.parent.mkdir()` then `open(path, "w")` then `json.dump(data, f)`. Replace with `self._store.save(self.MODEL_NAME, data)`. Directory creation is now the store's responsibility.

### Subtask T022 -- Verify rule-based fallback on load failure

- **Purpose**: Verify that all 5 model classes correctly fall back to rule-based / default predictions when `ModelStore.load()` returns `None`. This is spec requirement FR-011.

- **Steps**:
  1. Verify each model's fallback behavior matches current behavior:
     - `StuckPredictor`: `self.model = None`, `self._trained = False`. `predict()` returns `{"probability": 0.5, "confidence": "weak"}`.
     - `ActivityClassifier`: `self._ml_model = None`, `self._trained = False`. `classify()` delegates to `_classify_rules()`.
     - `WorkflowStatePredictor`: `self._ml_model = None`, `self._trained = False`. `predict()` delegates to `_predict_rules()`.
     - `DurationEstimator`: `self.model = None`, `self._trained = False`. `predict()` returns `{"estimated_minutes": 60.0, "confidence_interval": [30.0, 90.0]}`.
     - `QualityEstimator`: `self.weights = dict(DEFAULT_WEIGHTS)` (initialized before `_load_weights()`). `predict()` uses default weights.
  2. Verify the `_load()` method in each model:
     - Checks `if data is not None:` before attempting deserialization.
     - Catches deserialization exceptions with try/except.
     - Calls `evict()` on the store if deserialization fails (defensive, for cache).
     - Sets model to `None` / `False` on failure (activating rule-based fallback).
  3. Verify no exception propagates from `_load()` to callers -- the model simply starts untrained.

- **Files**:
  - Verify: All files in `src/sigil_ml/models/`

- **Parallel?**: No -- verify after T017-T021 are complete.

- **Notes**:
  - This is a verification subtask. The fallback logic already exists in current code. The refactor must preserve it exactly.
  - FR-011: "When a model cannot be loaded (missing, corrupted, backend unreachable), the system MUST fall back to rule-based predictions rather than returning an error."
  - Three failure modes: (1) `load()` returns `None` (missing model), (2) deserialization raises (corrupted data), (3) store is unreachable (handled by store returning `None`). All three must result in rule-based fallback.

## Risks & Mitigations

- **Risk**: `joblib.load(BytesIO(data))` behaves differently from `joblib.load(path)`. **Mitigation**: Both go through the same pickle machinery. `joblib` officially supports file-like objects. Byte representation is identical.
- **Risk**: Accidentally modifying prediction logic during refactor. **Mitigation**: `predict()` method bodies are NOT touched. Only `__init__`, `_load`, `_save`, and `train` are modified. Code review should verify no prediction code changed.
- **Risk**: Constructor signature change breaks callers (`Trainer`, `app.py`). **Mitigation**: Default `model_store=None` preserves backward compatibility. `StuckPredictor()` (no args) still works by creating a `LocalModelStore()` internally. WP05 updates callers to pass stores explicitly.
- **Risk**: Import cycle between models and storage. **Mitigation**: `TYPE_CHECKING` guard for the `ModelStore` import. The default-store creation (`from sigil_ml.storage import LocalModelStore`) is inside `__init__`, not at module level, avoiding circular import at module load time.
- **Risk**: `QualityEstimator` JSON round-trip issues. **Mitigation**: The weights dict is `{str: int}` -- trivially serializable with `json`. No edge cases.

## Review Guidance

For each of the 5 model classes, verify:
1. `__init__` accepts `model_store: "ModelStore | None" = None`.
2. When `None`, creates `LocalModelStore()` internally (backward compatible).
3. No `config.weights_path()` calls remain.
4. No `open()`, `Path.read_bytes()`, or `Path.write_bytes()` operations for weight files remain.
5. `_load()` handles `None` (missing model) correctly -- no deserialization attempted.
6. `_load()` handles deserialization errors with try/except -- falls back to untrained state.
7. `train()` / `_save()` serializes to bytes in memory, then calls `store.save()`.
8. `predict()` method is COMPLETELY unchanged (diff should show zero changes to prediction code).
9. Rule-based fallback still works when model is not loaded.

**Special attention**: `QualityEstimator` uses JSON, not joblib. Verify `json.loads(data.decode("utf-8"))` and `json.dumps(...).encode("utf-8")`.

## Implementation Command

```bash
spec-kitty implement WP04 --base WP01
```

## Activity Log

- 2026-03-30T01:45:11Z -- system -- lane=planned -- Prompt created.
