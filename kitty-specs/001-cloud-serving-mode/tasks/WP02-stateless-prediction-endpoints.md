---
work_package_id: WP02
title: Stateless Prediction Endpoints
lane: planned
dependencies: [WP01]
subtasks:
- T007
- T008
- T009
- T010
- T011
- T012
phase: Phase 2 - Core Endpoints
assignee: ''
agent: ''
shell_pid: ''
review_status: ''
reviewed_by: ''
history:
- timestamp: '2026-03-30T01:45:14Z'
  lane: planned
  agent: system
  shell_pid: ''
  action: Prompt regenerated via /spec-kitty.tasks
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

- All five `/predict/*` endpoints return valid predictions in cloud mode when features are provided in the request body.
- In cloud mode, `task_id`-only requests to `/predict/stuck` and `/predict/duration` return 400 with a descriptive error.
- In cloud mode, `/predict/suggest` requires `classified_events` in the request body (no poller buffer fallback).
- `/train` returns 405 Method Not Allowed in cloud mode.
- `/status` returns cloud-appropriate response without any SQLite queries.
- `/predict/quality` is verified to have no hidden SQLite dependency (already cloud-safe).
- In local mode, ALL endpoint behavior is identical to current implementation -- zero regression.

**Measurable**:
- POST to all 5 `/predict/*` endpoints with inline features returns 200 in cloud mode.
- POST to `/predict/stuck` with only `{"task_id": "abc"}` returns 400 in cloud mode.
- POST to `/predict/suggest` with empty body returns 400 in cloud mode.
- POST to `/train` returns 405 in cloud mode, 200 in local mode.
- GET `/status` returns valid JSON in cloud mode without SQLite errors.

## Context & Constraints

- **Spec**: FR-002 (no SQLite in cloud), FR-003 (serve all predict endpoints), FR-006 (local mode unchanged), FR-008 (no SQLite writes)
- **Plan**: Design Decision D6 (endpoint guards in cloud mode), D7 (create_app factory)
- **Research**: R6 (endpoint behavior in cloud mode)
- **Prerequisite**: WP01 must be complete. `AppState.mode` (type `ServingMode`) must be available on `state`.

**Current code you will modify** (`src/sigil_ml/routes.py`, 243 lines):
- `predict_stuck()` (line 146): Checks `req.features`, falls back to `extract_stuck_features(config.db_path(), req.task_id)` -- this calls `sqlite3.connect()`.
- `predict_suggest()` (line 161): Falls back to `state.poller._buffer` (line 184) when no `classified_events` -- poller is `None` in cloud mode.
- `predict_duration()` (line 190): Same SQLite pattern as stuck via `extract_duration_features()`.
- `predict_quality()` (line 205): Already requires `features` in body (`QualityRequest.features` is not optional).
- `train()` (line 221): Starts background training against SQLite.
- `status()` (line 120): Opens `sqlite3.connect(str(db), timeout=5.0)` directly.

**Import needed**: Add `HTTPException` to the existing `from fastapi import ...` line. Add `from sigil_ml.config import ServingMode`.

**Implementation command**: `spec-kitty implement WP02 --base WP01`

## Subtasks & Detailed Guidance

### Subtask T007 -- Refactor `/predict/stuck` for cloud mode

- **Purpose**: Prevent SQLite access when running in cloud mode. The `extract_stuck_features()` function (imported from `features.py`) calls `sqlite3.connect()` internally.
- **Steps**:
  1. In the `predict_stuck()` handler, add a cloud-mode guard in the `task_id` branch:
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
                     detail="Cloud mode requires 'features' in request body. "
                            "'task_id' lookup is not available without SQLite.",
                 )
             features = extract_stuck_features(config.db_path(), req.task_id)
         else:
             return StuckResponse(probability=0.5, confidence="weak")

         result = state.stuck.predict(features)
         return StuckResponse(**result)
     ```
  2. Add imports at the top of `routes.py`:
     ```python
     from fastapi import BackgroundTasks, FastAPI, HTTPException  # add HTTPException
     from sigil_ml.config import ServingMode
     ```
- **Files**: `src/sigil_ml/routes.py` (modify)
- **Parallel?**: No -- establishes the guard pattern used by T008 and T009.
- **Notes**: The `state.stuck is None` fallback still works in cloud mode (returns 0.5/weak). After WP05, `state.stuck` will be replaced by per-tenant model resolution. The guard here prevents the SQLite code path from executing.

### Subtask T008 -- Refactor `/predict/suggest` for cloud mode

- **Purpose**: The suggest endpoint falls back to `state.poller._buffer` (line 184 of current `routes.py`). In cloud mode, `state.poller` is `None`, so accessing `._buffer` would raise `AttributeError`.
- **Steps**:
  1. Update the `predict_suggest()` handler to guard the poller fallback:
     ```python
     @fastapi_app.post("/predict/suggest", response_model=WorkflowStateResponse)
     async def predict_suggest(req: WorkflowStateRequest) -> WorkflowStateResponse:
         if state.workflow is None:
             return WorkflowStateResponse(
                 flow_state={
                     "shallow_work": 1.0, "deep_work": 0.0,
                     "exploring": 0.0, "blocked": 0.0, "winding_down": 0.0,
                 },
                 dominant_state="shallow_work",
                 momentum=0.0, focus_score=0.5,
                 dominant_activity="idle", activity_distribution={},
                 session_elapsed_min=0.0, method="rules", confidence=0.5,
             )

         classified_events = req.classified_events or []
         session_info = {
             "session_elapsed_min": 0.0,
             "task_phase": None,
             "test_failures": 0,
         }

         if not classified_events:
             if state.mode == ServingMode.CLOUD:
                 raise HTTPException(
                     status_code=400,
                     detail="Cloud mode requires 'classified_events' in request body. "
                            "Poller buffer is not available.",
                 )
             elif state.poller:
                 classified_events = state.poller._buffer

         result = state.workflow.predict(classified_events, session_info)
         return WorkflowStateResponse(**result)
     ```
  2. This preserves the local-mode fallback to `state.poller._buffer` while returning a clear 400 error in cloud mode.
- **Files**: `src/sigil_ml/routes.py` (modify)
- **Parallel?**: No -- follows the pattern from T007.
- **Notes**: Returning 400 (rather than a fallback response) is intentional: the Go daemon calling this endpoint should always send classified events. An empty-events call from the Go daemon is a bug that should surface as an error.

### Subtask T009 -- Refactor `/predict/duration` for cloud mode

- **Purpose**: Same pattern as stuck -- `extract_duration_features()` calls `sqlite3.connect()` internally.
- **Steps**:
  1. Apply the same guard pattern as T007:
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
                     detail="Cloud mode requires 'features' in request body. "
                            "'task_id' lookup is not available without SQLite.",
                 )
             features = extract_duration_features(config.db_path(), req.task_id)
         else:
             return DurationResponse(estimated_minutes=60.0, confidence_interval=[30.0, 90.0])

         result = state.duration.predict(features)
         return DurationResponse(**result)
     ```
- **Files**: `src/sigil_ml/routes.py` (modify)
- **Parallel?**: No -- follows established pattern.

### Subtask T010 -- Audit `/predict/quality` for cloud safety

- **Purpose**: Verify that the quality endpoint has no hidden SQLite dependency.
- **Steps**:
  1. Review `predict_quality()` handler (line 205 of current `routes.py`):
     - `QualityRequest` has `features: dict[str, float]` as a **required** field (no default, no `None` option).
     - `state.quality.predict(req.features)` is the only call.
  2. Read `src/sigil_ml/models/quality.py` -- verify `QualityEstimator.predict()` does not import or call `sqlite3`.
  3. If confirmed safe, add a comment above the handler:
     ```python
     # Cloud-safe: QualityRequest.features is required (no task_id lookup path).
     # QualityEstimator.predict() is purely functional, no DB access.
     ```
  4. No code changes unless a hidden dependency is found.
- **Files**: `src/sigil_ml/routes.py` (audit + comment), `src/sigil_ml/models/quality.py` (audit only)
- **Parallel?**: Yes -- independent of other subtasks.

### Subtask T011 -- Guard `/train` endpoint: return 405 in cloud mode

- **Purpose**: Training against SQLite is not supported in cloud mode. Return a clear, appropriate error.
- **Steps**:
  1. Add a mode check at the top of the `train()` handler:
     ```python
     @fastapi_app.post("/train", response_model=TrainResponse)
     async def train(
         req: TrainRequest, background_tasks: BackgroundTasks
     ) -> TrainResponse:
         if state.mode == ServingMode.CLOUD:
             raise HTTPException(
                 status_code=405,
                 detail="Training is not supported in cloud mode. "
                        "Train models via the training pipeline and deploy weights to storage.",
             )

         if state.training_in_progress:
             return TrainResponse(status="busy", message="Training already in progress")

         db = req.db or str(config.db_path())
         background_tasks.add_task(_run_training, state, db)
         return TrainResponse(status="started", message=f"Training started with db={db}")
     ```
  2. 405 Method Not Allowed is the correct HTTP status: the operation exists but is not supported in this mode.
- **Files**: `src/sigil_ml/routes.py` (modify)
- **Parallel?**: Yes -- independent of prediction endpoint changes.

### Subtask T012 -- Update `/status` endpoint for cloud mode

- **Purpose**: The current `/status` handler (line 120) opens `sqlite3.connect(str(db), timeout=5.0)` directly. In cloud mode, this must be replaced with a SQLite-free response.
- **Steps**:
  1. Add a mode branch at the top of the `status()` handler:
     ```python
     @fastapi_app.get("/status")
     async def status() -> dict:
         if state.mode == ServingMode.CLOUD:
             return {
                 "mode": "cloud",
                 "poller_running": False,
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
                 "cursor": None,
                 "latest_predictions": [],
                 # Cache stats and tenant info added by WP06.
             }

         # Local mode: existing SQLite-based status (unchanged)
         db = config.db_path()
         try:
             conn = sqlite3.connect(str(db), timeout=5.0)
             # ... rest of existing code unchanged ...
     ```
  2. Include `cursor: None` and `latest_predictions: []` in the cloud response to maintain response shape compatibility with local mode.
  3. The cloud response is intentionally minimal for now. WP06 enriches it with cache stats and tenant info after WP05 is integrated.
- **Files**: `src/sigil_ml/routes.py` (modify)
- **Parallel?**: Yes -- independent of prediction endpoint changes.
- **Notes**: Do NOT remove the existing local-mode SQLite status code. It must continue to work exactly as before.

## Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Breaking local mode | Low | High | Every change adds a cloud-mode guard but leaves the `else` path identical. Run `pytest tests/test_server.py` after each change. |
| `sqlite3` import at module level in `features.py` | N/A | None | Module-level imports are fine -- only calling the functions would fail. The guards prevent the call path. |
| Missing `HTTPException` import | Low | Medium | Add to existing `from fastapi import ...` line at top of `routes.py`. |

## Review Guidance

- For each `/predict/*` endpoint, verify TWO paths:
  1. Cloud mode with features in body -> returns 200 with prediction.
  2. Cloud mode with only `task_id` -> returns 400 (stuck, duration) or missing events (suggest).
- Verify local mode is completely unchanged by running `pytest tests/test_server.py`.
- Verify `/train` returns 405 in cloud mode but still starts training in local mode.
- Verify `/status` returns a dict without any SQLite access in cloud mode.
- Verify `HTTPException` and `ServingMode` are imported correctly.

## Activity Log

- 2026-03-30T01:45:14Z -- system -- lane=planned -- Prompt regenerated.
