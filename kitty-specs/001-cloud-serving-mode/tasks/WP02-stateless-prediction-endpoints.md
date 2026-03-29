---
work_package_id: WP02
title: Stateless Prediction Endpoints
lane: planned
dependencies: [WP01]
subtasks:
- T006
- T007
- T008
- T009
- T010
- T011
phase: Phase 2 - Core Implementation
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
- FR-002
- FR-003
- FR-006
- FR-008
---

# Work Package Prompt: WP02 -- Stateless Prediction Endpoints

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

- All five `/predict/*` endpoints (stuck, suggest, duration, quality, plus the activity-related suggest) accept features directly in the request body and return valid predictions in cloud mode.
- No SQLite connection is opened during any cloud-mode prediction request.
- The `/train` endpoint returns an appropriate error in cloud mode.
- The `/status` endpoint returns cloud-appropriate information without querying SQLite.
- All existing local-mode behavior remains unchanged.

**Success gate**: Start `sigil-ml` with `--mode cloud` (no SQLite file on disk), POST features to each `/predict/*` endpoint, and receive valid JSON responses. Verify no `sqlite3` errors in logs.

## Context & Constraints

- **Spec**: FR-002 (no SQLite in cloud), FR-003 (serve all predict endpoints), FR-006 (local mode unchanged), FR-008 (no SQLite writes).
- **Dependency**: WP01 must be complete -- `AppState.mode` and `ServingMode` enum must be available.
- **Current routes.py behavior**: Some endpoints (stuck, duration) can accept features in the body OR look up by `task_id` from SQLite. In cloud mode, the SQLite path must be disabled.
- **The suggest endpoint** currently falls back to `state.poller._buffer` when no `classified_events` are provided. In cloud mode there is no poller.
- **Model availability**: After WP01, models are `None` in cloud mode (loaded on-demand in WP05). Endpoints must return fallback responses when the model is `None`.

## Subtasks & Detailed Guidance

### Subtask T006 -- Refactor `/predict/stuck` to always work with inline features in cloud mode

- **Purpose**: In cloud mode, the endpoint must not attempt SQLite access. Features must come from the request body.
- **Steps**:
  1. In `routes.py`, update the `predict_stuck()` handler:
     ```python
     @fastapi_app.post("/predict/stuck", response_model=StuckResponse)
     async def predict_stuck(req: StuckRequest) -> StuckResponse:
         if state.stuck is None:
             return StuckResponse(probability=0.5, confidence="weak")

         if req.features is not None:
             features = req.features
         elif req.task_id is not None:
             if state.mode == ServingMode.CLOUD:
                 raise HTTPException(
                     status_code=400,
                     detail="Cloud mode requires 'features' in request body. 'task_id' lookup is not available.",
                 )
             features = extract_stuck_features(config.db_path(), req.task_id)
         else:
             return StuckResponse(probability=0.5, confidence="weak")

         result = state.stuck.predict(features)
         return StuckResponse(**result)
     ```
  2. Add the necessary import at the top of `routes.py`:
     ```python
     from fastapi import BackgroundTasks, FastAPI, HTTPException
     from sigil_ml.config import ServingMode
     ```
  3. The `state.mode` attribute was added in WP01 (T003).
- **Files**: `src/sigil_ml/routes.py`
- **Parallel?**: No (establishes the pattern other endpoints follow).
- **Notes**: The existing fallback when `state.stuck is None` already handles the case where no model is loaded (cloud mode before WP05). The key change is rejecting `task_id`-only requests with a clear 400 error.

### Subtask T007 -- Refactor `/predict/suggest` to accept `classified_events` and `session_info` directly

- **Purpose**: The suggest endpoint currently falls back to `state.poller._buffer`. In cloud mode there is no poller, so the caller must provide classified events and session info.
- **Steps**:
  1. Update `WorkflowStateRequest` to make `session_info` an optional field:
     ```python
     class WorkflowStateRequest(BaseModel):
         task_id: str | None = None
         classified_events: list[dict] | None = None
         session_info: dict | None = None  # NEW: allow caller to provide session context
     ```
  2. Update the `predict_suggest()` handler:
     ```python
     @fastapi_app.post("/predict/suggest", response_model=WorkflowStateResponse)
     async def predict_suggest(req: WorkflowStateRequest) -> WorkflowStateResponse:
         if state.workflow is None:
             return WorkflowStateResponse(
                 flow_state={"shallow_work": 1.0, "deep_work": 0.0, "exploring": 0.0, "blocked": 0.0, "winding_down": 0.0},
                 dominant_state="shallow_work",
                 momentum=0.0, focus_score=0.5,
                 dominant_activity="idle", activity_distribution={},
                 session_elapsed_min=0.0, method="rules", confidence=0.5,
             )

         classified_events = req.classified_events or []
         session_info = req.session_info or {"session_elapsed_min": 0.0, "task_phase": None, "test_failures": 0}

         if not classified_events:
             if state.mode == ServingMode.CLOUD:
                 # In cloud mode, no poller buffer available
                 return WorkflowStateResponse(
                     flow_state={"shallow_work": 1.0, "deep_work": 0.0, "exploring": 0.0, "blocked": 0.0, "winding_down": 0.0},
                     dominant_state="shallow_work",
                     momentum=0.0, focus_score=0.5,
                     dominant_activity="idle", activity_distribution={},
                     session_elapsed_min=0.0, method="rules", confidence=0.5,
                 )
             elif state.poller:
                 classified_events = state.poller._buffer

         result = state.workflow.predict(classified_events, session_info)
         return WorkflowStateResponse(**result)
     ```
  3. Note: the fallback when no events are provided in cloud mode returns a rule-based response rather than an error, since an empty event buffer is a valid state.
- **Files**: `src/sigil_ml/routes.py`
- **Parallel?**: No (follows the pattern from T006).
- **Notes**: The `session_info` field is optional to maintain backward compatibility. In cloud mode, the Go daemon will send both `classified_events` and `session_info` in the request body.

### Subtask T008 -- Refactor `/predict/duration` to always work with inline features in cloud mode

- **Purpose**: Same pattern as stuck -- reject `task_id`-only requests in cloud mode.
- **Steps**:
  1. Update `predict_duration()` following the same pattern as T006:
     ```python
     @fastapi_app.post("/predict/duration", response_model=DurationResponse)
     async def predict_duration(req: DurationRequest) -> DurationResponse:
         if state.duration is None:
             return DurationResponse(estimated_minutes=60.0, confidence_interval=[30.0, 90.0])

         if req.features is not None:
             features = req.features
         elif req.task_id is not None:
             if state.mode == ServingMode.CLOUD:
                 raise HTTPException(
                     status_code=400,
                     detail="Cloud mode requires 'features' in request body. 'task_id' lookup is not available.",
                 )
             features = extract_duration_features(config.db_path(), req.task_id)
         else:
             return DurationResponse(estimated_minutes=60.0, confidence_interval=[30.0, 90.0])

         result = state.duration.predict(features)
         return DurationResponse(**result)
     ```
- **Files**: `src/sigil_ml/routes.py`
- **Parallel?**: No (follows the established pattern).
- **Notes**: Exact same refactor pattern as T006 applied to the duration endpoint.

### Subtask T009 -- Verify `/predict/quality` cloud safety

- **Purpose**: The quality endpoint already accepts features in the request body. Verify it has no hidden SQLite path.
- **Steps**:
  1. Review the current `predict_quality()` handler:
     ```python
     @fastapi_app.post("/predict/quality", response_model=QualityResponse)
     async def predict_quality(req: QualityRequest) -> QualityResponse:
         if state.quality is None:
             return QualityResponse(score=50, components={}, status="normal")
         result = state.quality.predict(req.features)
         return QualityResponse(score=result["score"], components=result["components"], status=result["status"])
     ```
  2. This endpoint already requires `features` in the body (`QualityRequest.features` is not optional). No SQLite access occurs. No changes needed unless the `QualityEstimator.predict()` method internally touches SQLite -- check `src/sigil_ml/models/quality.py`.
  3. If `quality.py` is clean (rule-based, no DB access), mark this subtask as verified with a code comment:
     ```python
     # Cloud-safe: QualityEstimator.predict() is purely functional, no DB access
     ```
- **Files**: `src/sigil_ml/routes.py`, `src/sigil_ml/models/quality.py` (read-only check)
- **Parallel?**: Yes (independent verification).
- **Notes**: This is a verification subtask. If quality.py does touch SQLite, refactor it to accept features only.

### Subtask T010 -- Guard the `/train` endpoint: return 405 in cloud mode

- **Purpose**: Training is a local-only operation. In cloud mode, the `/train` endpoint should return a clear error.
- **Steps**:
  1. Update the `train()` handler:
     ```python
     @fastapi_app.post("/train", response_model=TrainResponse)
     async def train(req: TrainRequest, background_tasks: BackgroundTasks) -> TrainResponse:
         if state.mode == ServingMode.CLOUD:
             raise HTTPException(
                 status_code=405,
                 detail="Training is not available in cloud mode. Use the training pipeline instead.",
             )

         if state.training_in_progress:
             return TrainResponse(status="busy", message="Training already in progress")

         db = req.db or str(config.db_path())
         background_tasks.add_task(_run_training, state, db)
         return TrainResponse(status="started", message=f"Training started with db={db}")
     ```
- **Files**: `src/sigil_ml/routes.py`
- **Parallel?**: Yes (independent of other endpoint changes).
- **Notes**: 405 Method Not Allowed is appropriate since the operation is not supported in this mode.

### Subtask T011 -- Guard the `/status` endpoint: return cloud-appropriate response

- **Purpose**: The `/status` endpoint currently queries SQLite directly. In cloud mode it must return useful information without DB access.
- **Steps**:
  1. Update the `status()` handler:
     ```python
     @fastapi_app.get("/status")
     async def status() -> dict:
         if state.mode == ServingMode.CLOUD:
             return {
                 "mode": "cloud",
                 "models_loaded": {
                     name: model is not None
                     for name, model in [
                         ("stuck", state.stuck),
                         ("activity", state.activity),
                         ("workflow", state.workflow),
                         ("duration", state.duration),
                         ("quality", state.quality),
                     ]
                 },
                 "poller_running": False,
                 "cursor": None,
                 "latest_predictions": [],
             }

         # Existing local-mode implementation below (unchanged)
         db = config.db_path()
         ...
     ```
  2. The cloud-mode response is a stub for now. WP06 will enhance it with tenant info, cache stats, etc.
- **Files**: `src/sigil_ml/routes.py`
- **Parallel?**: Yes (independent of prediction endpoint changes).
- **Notes**: Keep the response shape compatible with the local-mode response by including `cursor` and `latest_predictions` fields (as null/empty). This prevents consumers from breaking.

## Risks & Mitigations

- **Breaking local mode**: Every endpoint change must retain the existing `task_id` lookup path when `state.mode == ServingMode.LOCAL`. Use the guard pattern consistently: check mode only in the `task_id` branch, not the `features` branch.
- **Missing the activity endpoint**: There is no separate `/predict/activity` endpoint in the current codebase. Activity classification is internal to the poller. Confirm this is correct -- the spec mentions 5 endpoints: stuck, suggest, duration, activity, quality. If `/predict/activity` needs to exist, create it.
- **Import cycles**: Adding `ServingMode` import to `routes.py` should be safe since `config.py` has no route dependencies.

## Review Guidance

- For each `/predict/*` endpoint: send a request with `features` in body (should work in both modes). Send a request with only `task_id` (should work in local mode, return 400 in cloud mode).
- Verify `/train` returns 405 in cloud mode.
- Verify `/status` returns valid JSON in cloud mode without SQLite errors.
- Verify all local-mode behavior is completely unchanged by running the existing test suite.
- Check that `HTTPException` is imported from `fastapi`.

## Activity Log

- 2026-03-29T16:29:58Z -- system -- lane=planned -- Prompt created.
