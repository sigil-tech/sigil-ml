---
work_package_id: WP03
title: Tenant Context Middleware and Routing
lane: planned
dependencies: [WP01]
subtasks:
- T012
- T013
- T014
- T015
- T016
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
- FR-004
---

# Work Package Prompt: WP03 -- Tenant Context Middleware and Routing

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

- Create a `TenantContext` dataclass representing per-request tenant identity.
- Implement a FastAPI dependency that extracts the tenant ID from the `X-Sigil-Tenant` request header.
- In cloud mode, all `/predict/*` endpoints require a valid tenant header; missing header returns 401.
- In local mode, tenant extraction is bypassed and a sentinel local context is used.
- Tenant ID appears in structured log output for cloud-mode requests.

**Success gate**: Send a cloud-mode request with `X-Sigil-Tenant: acme-corp` header, confirm the tenant ID is accessible in the route handler. Send without the header, receive a 401.

## Context & Constraints

- **Spec**: FR-004 (read tenant identifier from incoming requests).
- **Dependency**: WP01 must be complete -- `ServingMode` and `AppState.mode` must be available.
- **API gateway assumption**: The upstream API gateway handles authentication and sets the `X-Sigil-Tenant` header. This WP only does header extraction, not authentication.
- **No heavyweight deps**: The tenant module must use only stdlib + FastAPI/Pydantic.

## Subtasks & Detailed Guidance

### Subtask T012 -- Create `TenantContext` dataclass in `src/sigil_ml/tenant.py`

- **Purpose**: Provide a typed container for per-request tenant information that route handlers can depend on.
- **Steps**:
  1. Create a new file `src/sigil_ml/tenant.py`:
     ```python
     """Tenant context for multi-tenant cloud serving."""

     from __future__ import annotations

     from dataclasses import dataclass

     # Sentinel for local mode -- no real tenancy.
     LOCAL_TENANT_ID = "local"


     @dataclass(frozen=True)
     class TenantContext:
         """Per-request tenant identity extracted from the API gateway.

         Attributes:
             tenant_id: Unique identifier for the tenant (from X-Sigil-Tenant header).
             tier: Service tier for the tenant (default "default"). Reserved for future
                   rate-limiting or feature-gating.
         """
         tenant_id: str
         tier: str = "default"

         @property
         def is_local(self) -> bool:
             """True when running in local mode (no real tenancy)."""
             return self.tenant_id == LOCAL_TENANT_ID

         def __str__(self) -> str:
             return f"tenant={self.tenant_id} tier={self.tier}"
     ```
  2. Use `frozen=True` to make the context immutable after creation (safety for concurrent requests).
- **Files**: `src/sigil_ml/tenant.py` (new file)
- **Parallel?**: No -- downstream subtasks depend on this.
- **Notes**: Keep this simple. The `tier` field is a forward-looking placeholder. It defaults to `"default"` and is not populated from headers yet.

### Subtask T013 -- Implement FastAPI dependency for tenant extraction

- **Purpose**: Create a reusable FastAPI `Depends()` callable that extracts tenant context from the request.
- **Steps**:
  1. Add the dependency function to `src/sigil_ml/tenant.py`:
     ```python
     import os
     from fastapi import Request, HTTPException

     TENANT_HEADER = os.environ.get("SIGIL_TENANT_HEADER", "X-Sigil-Tenant")


     def get_tenant_context(request: Request) -> TenantContext:
         """FastAPI dependency that extracts tenant context from the request.

         In cloud mode (determined by app state), the tenant header is required.
         In local mode, returns a sentinel local context.
         """
         from sigil_ml.config import ServingMode  # avoid circular import

         # Access app state to check mode
         state = request.app.state.app_state

         if state.mode == ServingMode.LOCAL:
             return TenantContext(tenant_id=LOCAL_TENANT_ID, tier="local")

         # Cloud mode: require tenant header
         tenant_id = request.headers.get(TENANT_HEADER)
         if not tenant_id or not tenant_id.strip():
             raise HTTPException(
                 status_code=401,
                 detail=f"Missing required header: {TENANT_HEADER}",
             )

         return TenantContext(tenant_id=tenant_id.strip())
     ```
  2. To make `state` accessible via `request.app.state`, update `create_app()` in `app.py` to store the AppState on the FastAPI app:
     ```python
     # In create_app(), after creating state:
     application.state.app_state = state
     ```
  3. This pattern avoids passing `state` through closures and makes it accessible from any dependency.
- **Files**: `src/sigil_ml/tenant.py`, `src/sigil_ml/app.py` (minor addition)
- **Parallel?**: No -- T014 depends on this.
- **Notes**: The deferred import of `ServingMode` avoids a potential circular import chain (`tenant` -> `config` -> ...). The `SIGIL_TENANT_HEADER` env var allows operators to customize the header name.

### Subtask T014 -- Wire tenant dependency into all `/predict/*` route handlers

- **Purpose**: Make tenant context available to each prediction endpoint for future model routing (WP05).
- **Steps**:
  1. Import the dependency in `routes.py`:
     ```python
     from fastapi import Depends
     from sigil_ml.tenant import TenantContext, get_tenant_context
     ```
  2. Add `tenant: TenantContext = Depends(get_tenant_context)` as a parameter to each predict handler:
     ```python
     @fastapi_app.post("/predict/stuck", response_model=StuckResponse)
     async def predict_stuck(
         req: StuckRequest,
         tenant: TenantContext = Depends(get_tenant_context),
     ) -> StuckResponse:
         # For now, tenant is available but not used for model routing.
         # WP05 will use tenant.tenant_id to look up tenant-specific models.
         ...
     ```
  3. Apply the same pattern to all five endpoints: `predict_stuck`, `predict_suggest`, `predict_duration`, `predict_quality`, and any others.
  4. Do NOT change the endpoint logic yet -- just accept the parameter. WP05 uses it for model routing.
- **Files**: `src/sigil_ml/routes.py`
- **Parallel?**: No (applies to all endpoints).
- **Notes**: Adding `Depends(get_tenant_context)` to the function signature means FastAPI will automatically call it and inject the result. In local mode, it returns the sentinel context; in cloud mode, it extracts from the header.

### Subtask T015 -- Return 401 when tenant header is missing in cloud mode

- **Purpose**: Enforce that cloud-mode requests always carry a tenant identifier.
- **Steps**:
  1. This is already handled by the `get_tenant_context()` dependency in T013 which raises `HTTPException(status_code=401)` when the header is missing.
  2. Verify the behavior:
     - Cloud mode + header present -> 200 with prediction
     - Cloud mode + header missing -> 401 with error message
     - Local mode + header present -> 200 (header ignored, local context used)
     - Local mode + header missing -> 200 (local context used)
  3. If the `/health` and `/status` endpoints should be accessible without a tenant header (they are operational, not tenant-scoped), ensure they do NOT have the `get_tenant_context` dependency. Only `/predict/*` endpoints require it.
- **Files**: `src/sigil_ml/routes.py`, `src/sigil_ml/tenant.py`
- **Parallel?**: Yes (verification task).
- **Notes**: Health and status endpoints must remain accessible without tenant headers for monitoring systems.

### Subtask T016 -- Add tenant ID to structured log output for cloud-mode requests

- **Purpose**: Enable operators to filter logs by tenant for debugging and audit.
- **Steps**:
  1. In each `/predict/*` handler (after the tenant dependency injection), add a log line:
     ```python
     if not tenant.is_local:
         logger.info("predict/stuck: tenant=%s", tenant.tenant_id)
     ```
  2. Alternatively, create a small logging helper in `tenant.py`:
     ```python
     def log_tenant_request(logger: logging.Logger, endpoint: str, tenant: TenantContext) -> None:
         """Log tenant context for cloud-mode requests."""
         if not tenant.is_local:
             logger.info("%s: tenant=%s tier=%s", endpoint, tenant.tenant_id, tenant.tier)
     ```
  3. Call it at the top of each predict handler:
     ```python
     log_tenant_request(logger, "predict/stuck", tenant)
     ```
- **Files**: `src/sigil_ml/routes.py`, `src/sigil_ml/tenant.py`
- **Parallel?**: Yes (independent of T015).
- **Notes**: Keep logging lightweight. Do not log the full request body (may contain sensitive features).

## Risks & Mitigations

- **Circular imports**: `tenant.py` imports from `config.py`. `routes.py` imports from `tenant.py`. `app.py` imports from `routes.py`. Ensure no back-edges. The deferred import in `get_tenant_context()` handles this.
- **Header injection**: In production, the API gateway must strip any user-provided `X-Sigil-Tenant` headers and set its own. This is an operational concern, not a code concern, but worth documenting.
- **Performance**: The dependency is called on every request. It is trivially fast (one header lookup + one dataclass construction).

## Review Guidance

- Verify cloud mode with valid tenant header returns 200 for all `/predict/*` endpoints.
- Verify cloud mode without tenant header returns 401 for `/predict/*` endpoints.
- Verify `/health` and `/status` work without a tenant header in both modes.
- Verify local mode ignores the tenant header entirely and still works.
- Check that `request.app.state.app_state` is set in `create_app()`.

## Activity Log

- 2026-03-29T16:29:58Z -- system -- lane=planned -- Prompt created.
