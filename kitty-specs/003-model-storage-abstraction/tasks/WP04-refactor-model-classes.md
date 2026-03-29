---
work_package_id: WP04
title: Refactor Model Classes to Use ModelStore
lane: planned
dependencies: [WP01]
subtasks:
- T016
- T017
- T018
- T019
- T020
- T021
phase: Phase 2 - Model Refactor
assignee: ''
agent: ''
shell_pid: ''
review_status: ''
reviewed_by: ''
history:
- timestamp: '2026-03-29T16:30:00Z'
  lane: planned
  agent: system
  shell_pid: ''
  action: Prompt generated via /spec-kitty.tasks
requirement_refs:
- FR-007
- FR-011
---

# Work Package Prompt: WP04 -- Refactor Model Classes to Use ModelStore

## Review Feedback

> **Populated by `/spec-kitty.review`** -- Reviewers add detailed feedback here when work needs changes.

*[This section is empty initially.]*

---

## Objectives & Success Criteria

- Refactor all 5 model classes to accept a `ModelStore` instance via constructor injection.
- Remove all direct filesystem I/O for weight persistence from model classes (no more `config.weights_path()`, no direct `joblib.load(path)`, no `open(path)`).
- Preserve identical prediction behavior -- only the load/save path changes.
- Ensure fallback to rule-based predictions when `ModelStore.load()` returns `None`.

**Success Criteria (SC-002 from spec)**:
- No model class directly imports filesystem I/O operations (`open()`, `pathlib`, `os.path`) for weight persistence.
- Each model's `__init__` accepts a `store` parameter of type `ModelStore`.
- `load()` + `BytesIO` deserialization replaces direct file reads.
- `save()` + `BytesIO` serialization replaces direct file writes.
- All existing prediction logic is untouched.
- Rule-based fallback behavior is identical: if no weights exist, models return default predictions.

## Context & Constraints

- **Spec**: `kitty-specs/003-model-storage-abstraction/spec.md` -- FR-007, FR-011.
- **Depends on WP01**: `ModelStore` protocol in `src/sigil_ml/storage/__init__.py`.
- **Current code locations**:
  - `src/sigil_ml/models/stuck.py` -- `StuckPredictor`: Uses `config.weights_path("stuck")` + `joblib.load(path)` / `joblib.dump(model, path)`.
  - `src/sigil_ml/models/activity.py` -- `ActivityClassifier`: Uses `config.weights_path("activity")` + `joblib.load(path)` / `joblib.dump(model, path)`.
  - `src/sigil_ml/models/workflow.py` -- `WorkflowStatePredictor`: Uses `config.weights_path("workflow")` + `joblib.load(path)` / `joblib.dump(model, path)`.
  - `src/sigil_ml/models/duration.py` -- `DurationEstimator`: Uses `config.weights_path("duration")` + `joblib.load(model, path)`.
  - `src/sigil_ml/models/quality.py` -- `QualityEstimator`: Uses `config.weights_path("quality")` + `json.load(f)` / `json.dump(data, f)` via `open()`.
- **Constraint**: `joblib` import stays in model classes but is used via `BytesIO` instead of file paths.
- **Constraint**: `QualityEstimator` uses JSON serialization, not joblib. The `ModelStore` still handles the raw bytes; the model class serializes/deserializes with `json`.
- **Constraint**: The `config` import can remain for non-persistence concerns, but `config.weights_path()` must not be called.

## Subtasks & Detailed Guidance

### Subtask T016 -- Refactor StuckPredictor to use ModelStore

- **Purpose**: Convert the first model class to use `ModelStore` as a reference pattern for the remaining four.

- **Steps**:
  1. Modify `src/sigil_ml/models/stuck.py`:
     - Change `__init__` signature to accept `store`:
       ```python
       from __future__ import annotations
       from io import BytesIO
       from typing import TYPE_CHECKING

       if TYPE_CHECKING:
           from sigil_ml.storage import ModelStore

       class StuckPredictor:
           MODEL_NAME = "stuck"

           def __init__(self, store: "ModelStore") -> None:
               self._store = store
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
                       if hasattr(self._store, 'evict'):
                           self._store.evict(self.MODEL_NAME)
                       self.model = None
       ```
     - Change `train()` to serialize to bytes and save via store:
       ```python
       def train(self, X: np.ndarray, y: np.ndarray) -> None:
           self.model = GradientBoostingClassifier(
               n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42,
           )
           self.model.fit(X, y)
           self._trained = True

           buf = BytesIO()
           joblib.dump(self.model, buf)
           self._store.save(self.MODEL_NAME, buf.getvalue())
           logger.info("Saved stuck model to store")
       ```
  2. Remove the `from sigil_ml import config` import (if no longer needed for anything else in the file).
  3. The `predict()` method is **completely untouched** -- only load/save changes.

- **Files**:
  - Update: `src/sigil_ml/models/stuck.py`

- **Parallel?**: No -- this is the reference implementation. Do this first, then T017-T020 in parallel.

- **Notes**:
  - `BytesIO(data)` wraps the raw bytes from the store so `joblib.load()` can read it as a file-like object.
  - `BytesIO()` for writing creates an in-memory buffer that `joblib.dump()` writes to; `.getvalue()` extracts the bytes.
  - The `MODEL_NAME` class constant replaces hardcoded string literals and makes the pattern reusable.
  - The `evict` call on deserialization failure is a defensive measure for when the store is a `ModelCache` (from WP03). For `LocalModelStore`, `hasattr(store, 'evict')` returns `False` and the call is skipped.

### Subtask T017 -- Refactor ActivityClassifier to use ModelStore

- **Purpose**: Apply the same pattern from T016 to the activity classifier.

- **Steps**:
  1. Modify `src/sigil_ml/models/activity.py`:
     - Add `store: "ModelStore"` parameter to `__init__`.
     - Replace weight loading:
       ```python
       MODEL_NAME = "activity"

       def __init__(self, store: "ModelStore") -> None:
           self._store = store
           self._ml_model = None
           self._trained = False
           self._load()

       def _load(self) -> None:
           data = self._store.load(self.MODEL_NAME)
           if data is not None:
               try:
                   self._ml_model = joblib.load(BytesIO(data))
                   self._trained = True
                   logger.info("Loaded activity classifier from store")
               except Exception:
                   logger.warning("Failed to load activity classifier, using rules")
                   if hasattr(self._store, 'evict'):
                       self._store.evict(self.MODEL_NAME)
                   self._ml_model = None
       ```
     - Update `_save()`:
       ```python
       def _save(self) -> None:
           if self._ml_model is not None:
               buf = BytesIO()
               joblib.dump(self._ml_model, buf)
               self._store.save(self.MODEL_NAME, buf.getvalue())
               logger.info("Saved activity classifier to store")
       ```
  2. Remove `from sigil_ml import config` if no longer needed.
  3. Leave `classify()`, `classify_batch()`, `_classify_rules()`, `_classify_ml()`, and `train()` logic untouched.

- **Files**:
  - Update: `src/sigil_ml/models/activity.py`

- **Parallel?**: Yes -- independent file from T016, T018, T019, T020.

- **Notes**:
  - The `train()` method calls `self._save()` internally. Once `_save()` uses the store, training automatically persists via the store.
  - All classification logic (rule-based and ML) is prediction-side and is NOT touched.

### Subtask T018 -- Refactor WorkflowStatePredictor to use ModelStore

- **Purpose**: Apply the same pattern to the workflow model.

- **Steps**:
  1. Modify `src/sigil_ml/models/workflow.py`:
     - Add `store: "ModelStore"` parameter to `__init__`.
     - Replace weight loading with the same `_load()` pattern:
       ```python
       MODEL_NAME = "workflow"

       def __init__(self, store: "ModelStore") -> None:
           self._store = store
           self._ml_model = None
           self._trained = False
           self._load()

       def _load(self) -> None:
           data = self._store.load(self.MODEL_NAME)
           if data is not None:
               try:
                   self._ml_model = joblib.load(BytesIO(data))
                   self._trained = True
                   logger.info("Loaded workflow model from store")
               except Exception:
                   logger.warning("Failed to load workflow model, using rules")
                   if hasattr(self._store, 'evict'):
                       self._store.evict(self.MODEL_NAME)
                   self._ml_model = None
       ```
     - Update `_save()`:
       ```python
       def _save(self) -> None:
           if self._ml_model is not None:
               buf = BytesIO()
               joblib.dump(self._ml_model, buf)
               self._store.save(self.MODEL_NAME, buf.getvalue())
               logger.info("Saved workflow model to store")
       ```
  2. Remove `from sigil_ml import config` import if no longer needed.
  3. Leave all prediction logic (`predict()`, `_predict_rules()`, `_predict_ml()`, static methods) untouched.

- **Files**:
  - Update: `src/sigil_ml/models/workflow.py`

- **Parallel?**: Yes -- independent file.

- **Notes**:
  - `WorkflowStatePredictor` has many static helper methods (`_activity_distribution`, `_compute_momentum`, `_compute_focus`). These are completely unrelated to persistence and must not be changed.

### Subtask T019 -- Refactor DurationEstimator to use ModelStore

- **Purpose**: Apply the same pattern to the duration model.

- **Steps**:
  1. Modify `src/sigil_ml/models/duration.py`:
     - Add `store: "ModelStore"` parameter to `__init__`.
     - Replace weight loading:
       ```python
       MODEL_NAME = "duration"

       def __init__(self, store: "ModelStore") -> None:
           self._store = store
           self.model: GradientBoostingRegressor | None = None
           self._trained = False
           self._load()

       def _load(self) -> None:
           data = self._store.load(self.MODEL_NAME)
           if data is not None:
               try:
                   self.model = joblib.load(BytesIO(data))
                   self._trained = True
                   logger.info("Loaded duration model from store")
               except Exception:
                   logger.warning("Failed to load duration model, starting fresh")
                   if hasattr(self._store, 'evict'):
                       self._store.evict(self.MODEL_NAME)
                   self.model = None
       ```
     - Update `train()`:
       ```python
       def train(self, X: np.ndarray, y: np.ndarray) -> None:
           self.model = GradientBoostingRegressor(
               n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42,
           )
           self.model.fit(X, y)
           self._trained = True

           buf = BytesIO()
           joblib.dump(self.model, buf)
           self._store.save(self.MODEL_NAME, buf.getvalue())
           logger.info("Saved duration model to store")
       ```
  2. Remove `from sigil_ml import config` import if no longer needed.

- **Files**:
  - Update: `src/sigil_ml/models/duration.py`

- **Parallel?**: Yes -- independent file.

- **Notes**:
  - `DurationEstimator.predict()` accesses `self.model.estimators_` for confidence intervals. This internal sklearn attribute is preserved by joblib byte-round-trip serialization.

### Subtask T020 -- Refactor QualityEstimator to use ModelStore

- **Purpose**: Apply the store pattern to the quality model. **This one is different** -- it uses JSON serialization instead of joblib.

- **Steps**:
  1. Modify `src/sigil_ml/models/quality.py`:
     - Add `store: "ModelStore"` parameter to `__init__`.
     - Replace `_load_weights()`:
       ```python
       MODEL_NAME = "quality"

       def __init__(self, store: "ModelStore") -> None:
           self._store = store
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
                   if hasattr(self._store, 'evict'):
                       self._store.evict(self.MODEL_NAME)
       ```
     - Replace `_save_weights()`:
       ```python
       def _save_weights(self) -> None:
           data = json.dumps({"weights": self.weights}).encode("utf-8")
           self._store.save(self.MODEL_NAME, data)
       ```
  2. Remove `from pathlib import Path` import (no longer needed).
  3. Remove `from sigil_ml import config` import if no longer needed.
  4. Keep the `json` import -- it's used for serialization (not file I/O).

- **Files**:
  - Update: `src/sigil_ml/models/quality.py`

- **Parallel?**: Yes -- independent file.

- **Notes**:
  - **Key difference**: `QualityEstimator` uses `json.loads(data.decode("utf-8"))` instead of `joblib.load(BytesIO(data))`. The store returns raw bytes regardless; the model class decides how to deserialize.
  - The current code saves as JSON with a `.joblib` extension (via `config.weights_path("quality")`). The `LocalModelStore` preserves this -- files are still named `quality.joblib`. The extension is cosmetic.
  - Remove the `path.parent.mkdir()` call from `_save_weights()` -- directory creation is now the store's responsibility.

### Subtask T021 -- Ensure rule-based fallback on load failure

- **Purpose**: Verify that all 5 model classes correctly fall back to rule-based / default predictions when `ModelStore.load()` returns `None`.

- **Steps**:
  1. Verify each model's fallback behavior:
     - `StuckPredictor`: Returns `{"probability": 0.5, "confidence": "weak"}` when `self.model is None`.
     - `ActivityClassifier`: Uses `_classify_rules()` when `self._trained is False`.
     - `WorkflowStatePredictor`: Uses `_predict_rules()` when `self._trained is False`.
     - `DurationEstimator`: Returns `{"estimated_minutes": 60.0, "confidence_interval": [30.0, 90.0]}` when `self.model is None`.
     - `QualityEstimator`: Uses `DEFAULT_WEIGHTS` when store returns `None` (weights dict is initialized to defaults before `_load_weights()`).
  2. Verify that `_load()` handles `None` from `store.load()` by skipping deserialization (not trying to deserialize `None`).
  3. Verify that deserialization errors (corrupted data) result in fallback mode, not exceptions propagated to callers.

- **Files**:
  - Verify: All files in `src/sigil_ml/models/`

- **Parallel?**: No -- must be verified after T016-T020.

- **Notes**:
  - This is a verification subtask. The fallback behavior already exists in the current code. The refactor must preserve it exactly.
  - The `_load()` pattern in T016-T020 checks `if data is not None:` before attempting deserialization -- this is the key guard.
  - FR-011 from the spec: "When a model cannot be loaded (missing, corrupted, backend unreachable), the system MUST fall back to rule-based predictions rather than returning an error."

## Risks & Mitigations

- **Risk**: `joblib.load(BytesIO(data))` behaves differently from `joblib.load(path)` for certain model types. **Mitigation**: `joblib` officially supports file-like objects. Both paths go through the same pickle machinery. The byte representation is identical.
- **Risk**: Accidentally modifying prediction logic during refactor. **Mitigation**: The `predict()` method body is not touched. Only `__init__`, `_load`, `_save`, and `train` are modified. Diff should show no changes to prediction code.
- **Risk**: Breaking the `Trainer` class which instantiates model classes. **Mitigation**: WP05 updates the Trainer. During WP04, the Trainer will need to be updated to pass a store -- this creates a temporary break that WP05 resolves. Alternatively, add a default store parameter: `store: "ModelStore | None" = None` and fall back to `LocalModelStore` if None.
- **Risk**: Import cycle between models and storage. **Mitigation**: Use `TYPE_CHECKING` guard for the `ModelStore` import. The runtime import is only the string annotation.

## Review Guidance

- For each model class, verify:
  1. `__init__` accepts `store: "ModelStore"` parameter.
  2. No direct `config.weights_path()` calls remain.
  3. No `open()` or `Path` operations for weight files remain.
  4. `_load()` handles `None` return (missing model) correctly.
  5. `_load()` handles deserialization errors with try/except.
  6. `train()` / `_save()` serializes to `BytesIO` and calls `store.save()`.
  7. `predict()` method is unchanged.
  8. Rule-based fallback still works when model is not loaded.
- Special attention to `QualityEstimator` -- it uses JSON, not joblib.

## Implementation Command

```bash
spec-kitty implement WP04 --base WP01
```

## Activity Log

- 2026-03-29T16:30:00Z -- system -- lane=planned -- Prompt created.
