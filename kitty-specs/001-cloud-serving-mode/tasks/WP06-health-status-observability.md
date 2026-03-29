---
work_package_id: WP06
title: Cloud-Aware Health, Status, and Observability
lane: planned
dependencies:
- WP04
subtasks:
- T028
- T029
- T030
- T031
- T032
phase: Phase 4 - Polish
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
- FR-007
---

# Work Package Prompt: WP06 -- Cloud-Aware Health, Status, and Observability

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

- The `/health` endpoint includes a `mode` field (`"local"` or `"cloud"`) in its response.
- In cloud mode, `/health` reports model availability without referencing SQLite or poller state.
- In cloud mode, `/status` lists loaded tenants, their model versions, and cache statistics.
- Per-tenant request counters are available for operational monitoring.
- Startup logs clearly indicate the operating mode.

**Success gate**: Start in cloud mode, call `GET /health`, verify `mode: "cloud"` in response. Call `GET /status`, verify tenant/cache information. Verify no SQLite references in cloud-mode responses.

## Context & Constraints

- **Spec**: FR-007 (health endpoint reflects operating mode and model availability).
- **Dependencies**: WP01 (mode awareness), WP04 (cache stats), WP05 (tenant model loading).
- **Backward compatibility**: Adding `mode` to `HealthResponse` must not break existing consumers. Use a default value.
- **No tenant header required**: Health and status endpoints are operational (not tenant-scoped) and must work without the `X-Sigil-Tenant` header.

## Subtasks & Detailed Guidance

### Subtask T028 -- Update `HealthResponse` schema to include `mode` field

- **Purpose**: Make the operating mode visible in health check responses.
- **Steps**:
  1. Update `HealthResponse` in `routes.py`:
     ```python
     class HealthResponse(BaseModel):
         status: str
         mode: str = "local"  # Default for backward compatibility
         models: dict[str, str]
         uptime_sec: float
     ```
  2. The `mode` field has a default of `"local"` so existing consumers that don't expect it can still parse the response.
- **Files**: `src/sigil_ml/routes.py`
- **Parallel?**: No -- T029 depends on this.
- **Notes**: Using a string field (not `ServingMode` enum) for the response avoids coupling the API schema to internal enums.

### Subtask T029 -- Update `/health` handler to report cloud-specific state

- **Purpose**: In cloud mode, the health response should reflect cloud-specific information.
- **Steps**:
  1. Update the `health()` handler:
     ```python
     @fastapi_app.get("/health", response_model=HealthResponse)
     async def health() -> HealthResponse:
         if state.mode == ServingMode.CLOUD:
             # Cloud mode: report cache-based model availability
             models_status: dict[str, str] = {}
             if state.model_cache:
                 tenants = state.model_cache.loaded_tenants()
                 if tenants:
                     # Report models as "cached" if any tenant has them loaded
                     all_models = set()
                     for model_list in tenants.values():
                         all_models.update(model_list)
                     for name in ["stuck", "activity", "workflow", "duration", "quality"]:
                         models_status[name] = "cached" if name in all_models else "on_demand"
                 else:
                     for name in ["stuck", "activity", "workflow", "duration", "quality"]:
                         models_status[name] = "on_demand"
             else:
                 for name in ["stuck", "activity", "workflow", "duration", "quality"]:
                     models_status[name] = "not_initialized"

             return HealthResponse(
                 status="ok",
                 mode="cloud",
                 models=models_status,
                 uptime_sec=round(time.time() - _start_time, 1),
             )

         # Local mode: existing behavior (unchanged)
         models_status = {}
         for name, model in [
             ("stuck", state.stuck),
             ("activity", state.activity),
             ("workflow", state.workflow),
             ("duration", state.duration),
         ]:
             if model is not None:
                 models_status[name] = "ready" if model.is_trained else "untrained"
             else:
                 models_status[name] = "not_loaded"

         models_status["quality"] = "ready" if state.quality is not None else "not_loaded"

         return HealthResponse(
             status="ok",
             mode="local",
             models=models_status,
             uptime_sec=round(time.time() - _start_time, 1),
         )
     ```
  2. Cloud mode model statuses:
     - `"cached"`: Model is currently in the cache for at least one tenant.
     - `"on_demand"`: Model is not cached but will be loaded on first request.
     - `"not_initialized"`: Cache/loader not yet initialized (should not happen after startup).
- **Files**: `src/sigil_ml/routes.py`
- **Parallel?**: No (modifies the health handler).
- **Notes**: The cloud-mode health check does not enumerate all tenants (that's what `/status` is for). It gives a high-level view of service readiness.

### Subtask T030 -- Rewrite `/status` handler for cloud mode

- **Purpose**: Provide detailed operational information in cloud mode: loaded tenants, model versions, cache statistics.
- **Steps**:
  1. Update the `status()` handler's cloud-mode branch (building on WP02's T011 stub):
     ```python
     @fastapi_app.get("/status")
     async def status() -> dict:
         if state.mode == ServingMode.CLOUD:
             cache_stats = state.model_cache.stats() if state.model_cache else {}
             loaded = state.model_cache.loaded_tenants() if state.model_cache else {}
             return {
                 "mode": "cloud",
                 "cache": cache_stats,
                 "loaded_tenants": loaded,
                 "request_counts": dict(_request_counters),  # from T031
                 "poller_running": False,
             }

         # Local mode: existing implementation (unchanged)
         db = config.db_path()
         ...
     ```
  2. The `loaded_tenants` value is a dict like `{"acme": ["stuck", "workflow"], "globex": ["stuck"]}`.
  3. The `cache` value includes hits, misses, evictions, hit_rate from `ModelCache.stats()`.
- **Files**: `src/sigil_ml/routes.py`
- **Parallel?**: No (builds on T029 pattern).
- **Notes**: This replaces the stub from WP02 T011 with real data from the cache.

### Subtask T031 -- Add per-tenant request counters

- **Purpose**: Enable operators to see which tenants are active and their request volumes.
- **Steps**:
  1. Add a simple in-memory counter at module level in `routes.py`:
     ```python
     from collections import defaultdict

     # Per-tenant request counters (reset on restart).
     _request_counters: dict[str, int] = defaultdict(int)
     ```
  2. Increment the counter in each `/predict/*` handler:
     ```python
     async def predict_stuck(req: StuckRequest, tenant: TenantContext = Depends(get_tenant_context)) -> StuckResponse:
         if not tenant.is_local:
             _request_counters[tenant.tenant_id] += 1
         ...
     ```
  3. Alternatively, create a helper:
     ```python
     def _count_request(tenant: TenantContext) -> None:
         if not tenant.is_local:
             _request_counters[tenant.tenant_id] += 1
     ```
  4. Expose in the `/status` response (done in T030).
- **Files**: `src/sigil_ml/routes.py`
- **Parallel?**: Yes (independent utility).
- **Notes**: `defaultdict(int)` is not strictly thread-safe for reads+writes, but Python's GIL makes single-operation increments safe in practice. For a production system, consider `threading.Lock` or atomic counters. For MVP, this is acceptable.

### Subtask T032 -- Add `mode` to startup and structured log output

- **Purpose**: Make the operating mode immediately visible in logs for debugging and monitoring.
- **Steps**:
  1. The startup log line was partially addressed in WP01 T004. Ensure it clearly states the mode:
     ```python
     # In create_app() startup_event:
     logger.info("sigil-ml: started in %s mode", state.mode.value)
     ```
  2. Add mode to the application metadata:
     ```python
     application = FastAPI(
         title="sigil-ml",
         version="0.1.0",
         description=f"Sigil ML sidecar ({mode.value} mode)",
     )
     ```
  3. Consider adding a `/` root endpoint for quick identification:
     ```python
     @fastapi_app.get("/")
     async def root() -> dict:
         return {"service": "sigil-ml", "mode": state.mode.value, "version": "0.1.0"}
     ```
- **Files**: `src/sigil_ml/app.py`, `src/sigil_ml/routes.py`
- **Parallel?**: Yes (independent of other subtasks).
- **Notes**: This is a small polish subtask. The root endpoint is optional but useful for operators who curl the service to check what's running.

## Risks & Mitigations

- **HealthResponse schema change**: Adding `mode` with a default value is backward compatible. Consumers that don't parse `mode` will continue to work.
- **Request counter memory**: In a long-running service with many tenants, the counter dict could grow. Since it resets on restart and tenants are typically bounded, this is acceptable. For extreme cases, add a max-tenants cap.
- **Status endpoint information leak**: The `/status` endpoint reveals tenant IDs and request counts. In production, ensure it is not publicly accessible (K8s service-internal only).

## Review Guidance

- Verify `GET /health` returns `mode: "cloud"` in cloud mode and `mode: "local"` in local mode.
- Verify `GET /health` does NOT reference SQLite or poller in cloud mode.
- Verify `GET /status` returns cache stats and loaded tenants in cloud mode.
- Verify `GET /status` returns the existing SQLite-based response in local mode (no regression).
- Verify `/health` and `/status` work without `X-Sigil-Tenant` header in both modes.
- Verify request counters increment per tenant in cloud mode.

## Activity Log

- 2026-03-29T16:29:58Z -- system -- lane=planned -- Prompt created.
