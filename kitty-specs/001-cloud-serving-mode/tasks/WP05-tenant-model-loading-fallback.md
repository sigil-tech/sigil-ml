---
work_package_id: WP05
title: Tenant-Aware Model Loading and Fallback
lane: planned
dependencies:
- WP04
subtasks:
- T022
- T023
- T024
- T025
- T026
- T027
phase: Phase 3 - Integration
assignee: ''
agent: ''
shell_pid: ''
review_status: ''
reviewed_by: ''
history:
- timestamp: '2026-03-29T16:29:58Z'
  lane: planned
  agent: system
  shell_pid: ''
  action: Prompt generated via /spec-kitty.tasks
requirement_refs:
- FR-004
- FR-005
- FR-009
- FR-010
---

# Work Package Prompt: WP05 -- Tenant-Aware Model Loading and Fallback

## Review Feedback Status

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

## Objectives & Success Criteria

- Define a `ModelLoader` protocol for pluggable storage backends.
- Implement `FilesystemModelLoader` as the initial concrete implementation.
- Integrate `ModelCache` + `ModelLoader` into `AppState` for cloud-mode model resolution.
- Update each `/predict/*` handler to resolve tenant-specific models via the cache in cloud mode.
- Implement rule-based fallback predictions for all five model types when no trained model exists.
- Add structured logging for cache hit/miss/fallback events.

**Success gate**: Send prediction requests for tenant-A and tenant-B, each receives predictions from their respective cached models. Send a request for unknown-tenant, receive a rule-based fallback response (not an error).

## Context & Constraints

- **Spec**: FR-004 (tenant-specific model weights), FR-005 (rule-based fallback), FR-009 (pluggable storage backend), FR-010 (cache with TTL).
- **Dependencies**: WP02 (stateless endpoints), WP03 (TenantContext), WP04 (ModelCache).
- **Feature 003 (Model Storage Abstraction)**: Will provide an S3-backed `ModelLoader` implementation in the future. The `ModelLoader` protocol defined here must be compatible.
- **Model types**: `"stuck"`, `"activity"`, `"workflow"`, `"duration"`, `"quality"` -- these are the five model names used throughout the codebase.
- **Current model loading**: Each model class (e.g., `StuckPredictor`) loads weights from `config.weights_path(name)` in its `__init__`. In cloud mode, we need to load from a tenant-specific path instead.

## Subtasks & Detailed Guidance

### Subtask T022 -- Create `ModelLoader` protocol in `src/sigil_ml/loader.py`

- **Purpose**: Define a pluggable interface for loading model weights from any storage backend.
- **Steps**:
  1. Create `src/sigil_ml/loader.py`:
     ```python
     """Model loading interface for pluggable storage backends.

     Defines the ModelLoader protocol that storage backends must implement.
     Feature 003 (Model Storage Abstraction) will provide an S3 implementation.
     """

     from __future__ import annotations

     import logging
     from typing import Any, Protocol, runtime_checkable

     logger = logging.getLogger(__name__)


     @runtime_checkable
     class ModelLoader(Protocol):
         """Protocol for loading model objects from a storage backend.

         Implementations must handle:
         - Tenant-specific model resolution
         - Returning None when no model exists (not raising exceptions)
         - Thread-safe loading (may be called concurrently)
         """

         def load(self, tenant_id: str, model_name: str) -> Any | None:
             """Load a model for the given tenant and model name.

             Args:
                 tenant_id: The tenant identifier.
                 model_name: One of "stuck", "activity", "workflow", "duration", "quality".

             Returns:
                 The loaded model object, or None if no model exists for this tenant/name.
             """
             ...
     ```
  2. Use `@runtime_checkable` so we can use `isinstance(loader, ModelLoader)` for validation.
- **Files**: `src/sigil_ml/loader.py` (new file)
- **Parallel?**: No -- T023 implements this.
- **Notes**: The protocol is intentionally minimal. Storage backends only need to implement one method. Error handling (corrupted files, network failures) should be caught internally and logged, returning `None`.

### Subtask T023 -- Implement `FilesystemModelLoader`

- **Purpose**: Provide an initial concrete implementation that loads model weights from the local filesystem, organized by tenant.
- **Steps**:
  1. Add to `src/sigil_ml/loader.py`:
     ```python
     import joblib
     from pathlib import Path
     from sigil_ml import config


     class FilesystemModelLoader:
         """Loads model weights from the local filesystem.

         Directory structure:
             {models_dir}/{tenant_id}/{model_name}.joblib

         For local mode compatibility, also supports:
             {models_dir}/{model_name}.joblib (no tenant subdirectory)
         """

         def __init__(self, base_dir: Path | None = None) -> None:
             self._base_dir = base_dir or config.models_dir()

         def load(self, tenant_id: str, model_name: str) -> Any | None:
             """Load a model from the filesystem.

             Returns the raw sklearn model object (not the wrapper class), or None.
             """
             # Try tenant-specific path first
             tenant_path = self._base_dir / tenant_id / f"{model_name}.joblib"
             if tenant_path.exists():
                 try:
                     model = joblib.load(tenant_path)
                     logger.info("loader: loaded %s/%s from %s", tenant_id, model_name, tenant_path)
                     return model
                 except Exception:
                     logger.warning("loader: failed to load %s", tenant_path, exc_info=True)
                     return None

             # Fallback: shared model (no tenant directory)
             shared_path = self._base_dir / f"{model_name}.joblib"
             if shared_path.exists():
                 try:
                     model = joblib.load(shared_path)
                     logger.info("loader: loaded shared %s from %s", model_name, shared_path)
                     return model
                 except Exception:
                     logger.warning("loader: failed to load %s", shared_path, exc_info=True)
                     return None

             logger.debug("loader: no model found for %s/%s", tenant_id, model_name)
             return None
     ```
  2. The dual-path lookup (tenant-specific then shared) enables gradual rollout: deploy a shared model first, then override per-tenant as needed.
- **Files**: `src/sigil_ml/loader.py`
- **Parallel?**: No -- builds on T022.
- **Notes**: The loader returns raw sklearn model objects. The prediction endpoints already know how to use them (via the wrapper classes). WP05 T025 handles the integration layer.

### Subtask T024 -- Integrate `ModelCache` + `ModelLoader` into `AppState` for cloud mode

- **Purpose**: Wire cache and loader into the application state so prediction handlers can resolve models.
- **Steps**:
  1. Update `AppState` in `src/sigil_ml/app.py`:
     ```python
     from sigil_ml.cache import ModelCache, create_model_cache
     from sigil_ml.loader import FilesystemModelLoader, ModelLoader

     class AppState:
         def __init__(self, mode: ServingMode = ServingMode.LOCAL) -> None:
             self.mode = mode
             # Existing fields for local mode
             self.stuck: StuckPredictor | None = None
             self.activity: ActivityClassifier | None = None
             self.workflow: WorkflowStatePredictor | None = None
             self.duration: DurationEstimator | None = None
             self.quality: QualityEstimator | None = None
             self.poller: EventPoller | None = None
             self.training_in_progress: bool = False
             # Cloud mode fields
             self.model_cache: ModelCache | None = None
             self.model_loader: ModelLoader | None = None
     ```
  2. In `create_app()`, initialize cloud components during startup:
     ```python
     @application.on_event("startup")
     async def startup_event() -> None:
         if state.mode == ServingMode.LOCAL:
             # ... existing local startup (unchanged) ...
         else:
             # Cloud mode
             state.model_cache = create_model_cache()
             state.model_loader = FilesystemModelLoader()
             logger.info("sigil-ml: cloud mode — cache and loader initialized")
     ```
  3. Add a helper method to `AppState` for resolving models:
     ```python
     def resolve_model(self, tenant_id: str, model_name: str) -> Any | None:
         """Resolve a model for the given tenant, using cache then loader.

         Returns the model object or None if no model is available.
         Only used in cloud mode.
         """
         if self.model_cache is None or self.model_loader is None:
             return None

         # Check cache first
         model = self.model_cache.get(tenant_id, model_name)
         if model is not None:
             return model

         # Cache miss: load from backend
         model = self.model_loader.load(tenant_id, model_name)
         if model is not None:
             self.model_cache.put(tenant_id, model_name, model)
             return model

         return None
     ```
- **Files**: `src/sigil_ml/app.py`
- **Parallel?**: No -- T025 depends on this.
- **Notes**: The `resolve_model()` method encapsulates the cache-then-load pattern. Prediction handlers call this instead of accessing `state.stuck` etc. directly in cloud mode.

### Subtask T025 -- Update each `/predict/*` handler to resolve model via cache in cloud mode

- **Purpose**: Complete the cloud model serving loop: request arrives -> extract tenant -> resolve model from cache -> predict -> return.
- **Steps**:
  1. Update `predict_stuck()` as the reference implementation:
     ```python
     @fastapi_app.post("/predict/stuck", response_model=StuckResponse)
     async def predict_stuck(
         req: StuckRequest,
         tenant: TenantContext = Depends(get_tenant_context),
     ) -> StuckResponse:
         log_tenant_request(logger, "predict/stuck", tenant)

         if state.mode == ServingMode.CLOUD:
             # Cloud mode: resolve tenant-specific model
             model = state.resolve_model(tenant.tenant_id, "stuck")
             if model is None:
                 # Fallback: rule-based prediction
                 return StuckResponse(probability=0.5, confidence="weak")

             if req.features is None:
                 raise HTTPException(status_code=400, detail="Cloud mode requires 'features' in request body.")

             # Use the raw sklearn model directly
             import numpy as np
             from sigil_ml.models.stuck import FEATURE_NAMES
             x = np.array([[req.features.get(f, 0.0) for f in FEATURE_NAMES]])
             prob = float(model.predict_proba(x)[0, 1])
             confidence = "weak" if prob < 0.4 else ("moderate" if prob < 0.7 else "strong")
             return StuckResponse(probability=round(prob, 4), confidence=confidence)

         # Local mode: unchanged
         if state.stuck is None:
             return StuckResponse(probability=0.5, confidence="weak")

         if req.features is not None:
             features = req.features
         elif req.task_id is not None:
             features = extract_stuck_features(config.db_path(), req.task_id)
         else:
             return StuckResponse(probability=0.5, confidence="weak")

         result = state.stuck.predict(features)
         return StuckResponse(**result)
     ```
  2. **IMPORTANT**: Consider creating a helper function to avoid duplicating the raw-model-to-prediction logic:
     ```python
     def _predict_with_raw_model(model: Any, features: dict, feature_names: list[str]) -> dict:
         """Run prediction using a raw sklearn model loaded from cache."""
         import numpy as np
         x = np.array([[features.get(f, 0.0) for f in feature_names]])
         prob = float(model.predict_proba(x)[0, 1])
         confidence = "weak" if prob < 0.4 else ("moderate" if prob < 0.7 else "strong")
         return {"probability": round(prob, 4), "confidence": confidence}
     ```
  3. Apply the same pattern to all five endpoints:
     - **stuck**: Use `FEATURE_NAMES` from `models/stuck.py`, call `predict_proba`.
     - **suggest (workflow)**: Use `state.resolve_model(tenant_id, "workflow")`, call `predict(classified_events, session_info)`. If the loaded model is a raw sklearn model, use the `WorkflowStatePredictor` wrapper or call the model directly.
     - **duration**: Use `FEATURE_NAMES` from `models/duration.py`.
     - **quality**: `QualityEstimator` is rule-based (no sklearn model). In cloud mode, use a `QualityEstimator()` instance directly (no tenant-specific weights needed).
     - **activity**: Activity classification is internal (no endpoint currently). Not needed here.
  4. **Alternative approach** (simpler): Instead of extracting raw sklearn models, the `ModelLoader` could return fully initialized wrapper objects (`StuckPredictor`, etc.) with their sklearn model set. This avoids duplicating prediction logic. Evaluate which approach is cleaner.
- **Files**: `src/sigil_ml/routes.py`, potentially `src/sigil_ml/app.py`
- **Parallel?**: No (modifies all endpoints).
- **Notes**: The choice between raw sklearn models vs. wrapper objects is an implementation decision. If using wrapper objects, the `FilesystemModelLoader.load()` would need to know about the wrapper classes. If using raw models, the prediction logic needs to be available in `routes.py`. The wrapper approach is cleaner but couples the loader to model classes. Recommend the wrapper approach for maintainability.

### Subtask T026 -- Implement rule-based fallback predictions for all five model types

- **Purpose**: FR-005 requires graceful fallback when no trained model exists for a tenant.
- **Steps**:
  1. Create a centralized fallback module or add to `routes.py`:
     ```python
     # Fallback predictions for cloud mode when no model is available.
     # These match the existing fallbacks in poller.py and routes.py.

     FALLBACK_STUCK = StuckResponse(probability=0.5, confidence="weak")

     FALLBACK_SUGGEST = WorkflowStateResponse(
         flow_state={"shallow_work": 1.0, "deep_work": 0.0, "exploring": 0.0, "blocked": 0.0, "winding_down": 0.0},
         dominant_state="shallow_work",
         momentum=0.0,
         focus_score=0.5,
         dominant_activity="idle",
         activity_distribution={},
         session_elapsed_min=0.0,
         method="rules",
         confidence=0.5,
     )

     FALLBACK_DURATION = DurationResponse(estimated_minutes=60.0, confidence_interval=[30.0, 90.0])

     FALLBACK_QUALITY = QualityResponse(score=50, components={}, status="normal")
     ```
  2. Use these constants in each endpoint's cloud-mode branch when `state.resolve_model()` returns `None`.
  3. Ensure each fallback response includes `method: "rules"` or `confidence: "weak"` so consumers know it is not a trained-model prediction.
- **Files**: `src/sigil_ml/routes.py`
- **Parallel?**: Yes (independent constants, can be defined alongside T025).
- **Notes**: These fallbacks already exist scattered across `routes.py` and `poller.py`. Centralizing them avoids drift and makes them reusable.

### Subtask T027 -- Structured logging for model load events

- **Purpose**: Operators need visibility into cache behavior for debugging latency and model freshness issues.
- **Steps**:
  1. Add logging in `AppState.resolve_model()`:
     ```python
     def resolve_model(self, tenant_id: str, model_name: str) -> Any | None:
         if self.model_cache is None or self.model_loader is None:
             return None

         model = self.model_cache.get(tenant_id, model_name)
         if model is not None:
             logger.debug("model-resolve: cache_hit tenant=%s model=%s", tenant_id, model_name)
             return model

         model = self.model_loader.load(tenant_id, model_name)
         if model is not None:
             self.model_cache.put(tenant_id, model_name, model)
             logger.info("model-resolve: cache_miss+loaded tenant=%s model=%s", tenant_id, model_name)
             return model

         logger.info("model-resolve: cache_miss+fallback tenant=%s model=%s", tenant_id, model_name)
         return None
     ```
  2. Use `debug` level for cache hits (high frequency) and `info` level for cache misses (important for debugging).
- **Files**: `src/sigil_ml/app.py`
- **Parallel?**: Yes (logging additions are independent).
- **Notes**: Do not log the model object itself (it may be large). Log tenant_id and model_name only.

## Risks & Mitigations

- **Thundering herd on cache miss**: Multiple requests for the same tenant/model arriving simultaneously could all trigger loads. Mitigate with a per-key loading lock in `resolve_model()`:
  ```python
  self._loading_locks: dict[tuple[str, str], threading.Lock] = {}
  ```
  This is a nice-to-have optimization; the basic implementation works without it.
- **Model format incompatibility**: If a tenant's model was trained with a different scikit-learn version, `joblib.load` may fail or produce incorrect results. The `FilesystemModelLoader` catches exceptions and returns `None`.
- **Memory pressure**: Each loaded model consumes memory. The `ModelCache.max_size` (default 100) bounds this. With 5 models per tenant, this supports ~20 tenants. Adjust via env var for larger deployments.

## Review Guidance

- Verify the request flow: request -> tenant extraction -> model resolution (cache -> loader -> fallback) -> prediction -> response.
- Verify fallback responses match the existing defaults in the codebase.
- Verify local mode is completely unchanged (no `resolve_model` calls, no cache usage).
- Verify that model loading failures (corrupt files, missing files) result in fallback responses, not 500 errors.
- Verify structured logs show cache hit/miss/fallback events.
- Check that `FilesystemModelLoader` handles both tenant-specific and shared model paths.

## Activity Log

- 2026-03-29T16:29:58Z -- system -- lane=planned -- Prompt created.
