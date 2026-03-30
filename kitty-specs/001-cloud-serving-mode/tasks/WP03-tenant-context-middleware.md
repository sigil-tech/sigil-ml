---
work_package_id: WP03
title: Tenant Context Extraction and Routing
lane: planned
dependencies: [WP01]
subtasks:
- T013
- T014
- T015
- T016
- T017
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
- FR-004
---

# Work Package Prompt: WP03 -- Tenant Context Extraction and Routing

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

- Create `TenantContext` frozen dataclass with `tenant_id` and `tier` fields.
- Implement a FastAPI dependency that extracts tenant ID from the `X-Tenant-ID` request header.
- In cloud mode, missing header returns 401 Unauthorized.
- In local mode, the dependency returns a sentinel `TenantContext(tenant_id="local", tier="local")` without checking any header.
- Wire the tenant dependency into all `/predict/*` route handler signatures.
- Header name is configurable via `SIGIL_TENANT_HEADER` env var (default `X-Tenant-ID`).

**Measurable**:
- Cloud mode request with `X-Tenant-ID: tenant-abc` -> handler receives `TenantContext(tenant_id="tenant-abc")`.
- Cloud mode request without the header -> 401 response with descriptive error.
- Local mode request without any header -> request succeeds, handler receives sentinel context.
- `/health` and `/status` endpoints work without tenant header in both modes.

## Context & Constraints

- **Spec**: FR-004 (read tenant identifier from incoming requests)
- **Plan**: Design Decision D3 (tenant header name and behavior)
- **Data Model**: `data-model.md` -- TenantContext entity definition
- **Research**: R2 (tenant identification mechanism -- header extraction, no JWT)
- **Prerequisite**: WP01 must be complete. `AppState.mode` must be available.

**Key constraints**:
- Do NOT implement authentication. The API gateway handles that. This is purely header extraction.
- In local mode, the tenant context is created but never used for model routing (WP05 handles that).
- The tenant dependency applies only to `/predict/*` endpoints, NOT to `/health`, `/status`, `/plugins`, or `/train`.

**Implementation command**: `spec-kitty implement WP03 --base WP01`

## Subtasks & Detailed Guidance

### Subtask T013 -- Create `TenantContext` dataclass in `src/sigil_ml/tenant.py`

- **Purpose**: Define the per-request tenant context that downstream handlers and the model cache (WP05) use for routing.
- **Steps**:
  1. Create new file `src/sigil_ml/tenant.py`:
     ```python
     """Tenant context extraction for multi-tenant cloud serving."""

     from __future__ import annotations

     import logging
     import os
     from dataclasses import dataclass

     logger = logging.getLogger(__name__)

     # Sentinel for local mode -- not used for model routing.
     LOCAL_TENANT_ID = "local"


     @dataclass(frozen=True)
     class TenantContext:
         """Per-request tenant context extracted from the authenticated request.

         In cloud mode, populated from the X-Tenant-ID header (or configured header).
         In local mode, returns a sentinel with tenant_id="local".
         """

         tenant_id: str
         tier: str = "default"

         @property
         def is_local(self) -> bool:
             """True when running in local mode (no real tenancy)."""
             return self.tenant_id == LOCAL_TENANT_ID
     ```
  2. Use `frozen=True` for immutability -- tenant context should not be modified after extraction.
  3. `is_local` property allows handlers to easily check if tenant routing is active.
- **Files**: `src/sigil_ml/tenant.py` (new file, ~25 lines)
- **Parallel?**: No -- foundational for T014-T017.
- **Notes**: This is a plain dataclass, NOT a Pydantic model. It is internal routing context, not a request/response schema.

### Subtask T014 -- Implement FastAPI dependency function `get_tenant_context()`

- **Purpose**: Create a reusable FastAPI dependency that extracts tenant context from the request, respecting the serving mode.
- **Steps**:
  1. Add the dependency function and a factory to `src/sigil_ml/tenant.py`:
     ```python
     from typing import TYPE_CHECKING

     from fastapi import HTTPException, Request

     if TYPE_CHECKING:
         from sigil_ml.app import AppState


     def tenant_header_name() -> str:
         """Return the configured tenant header name."""
         return os.environ.get("SIGIL_TENANT_HEADER", "X-Tenant-ID")


     def make_tenant_dependency(state: "AppState"):
         """Create a FastAPI dependency that extracts tenant context.

         Uses a closure to capture the application state so the dependency
         can check the serving mode.

         Args:
             state: Application state containing the serving mode.

         Returns:
             An async callable suitable for FastAPI Depends().
         """
         from sigil_ml.config import ServingMode

         async def get_tenant_context(request: Request) -> TenantContext:
             """Extract tenant context from the request.

             In cloud mode: reads X-Tenant-ID header, returns 401 if missing.
             In local mode: returns sentinel context without checking headers.
             """
             if state.mode == ServingMode.LOCAL:
                 return TenantContext(tenant_id=LOCAL_TENANT_ID, tier="local")

             header = tenant_header_name()
             tenant_id = request.headers.get(header)
             if not tenant_id or not tenant_id.strip():
                 raise HTTPException(
                     status_code=401,
                     detail=f"Missing required header '{header}' for cloud mode.",
                 )
             return TenantContext(tenant_id=tenant_id.strip())

         return get_tenant_context
     ```
  2. The closure pattern (`make_tenant_dependency(state)`) captures `state` so the dependency has access to the serving mode without needing `request.app.state`.
  3. The deferred import of `ServingMode` inside the factory avoids potential circular import chains.
- **Files**: `src/sigil_ml/tenant.py` (modify -- add ~30 lines)
- **Parallel?**: No -- depends on T013, used by T015.
- **Notes**: Alternative approach: access `request.app.state.app_state` directly. The closure approach is chosen because it is more explicit and testable (you can pass a mock state).

### Subtask T015 -- Wire tenant dependency into all `/predict/*` route handler signatures

- **Purpose**: Make tenant context available in every prediction handler so WP05 can use it for model routing.
- **Steps**:
  1. In `src/sigil_ml/routes.py`, update `register_routes()` to create the tenant dependency:
     ```python
     from fastapi import Depends
     from sigil_ml.tenant import TenantContext, make_tenant_dependency

     def register_routes(fastapi_app: FastAPI, state: AppState) -> None:
         get_tenant = make_tenant_dependency(state)

         # ... all existing route definitions below, with updated signatures
     ```
  2. Add `tenant: TenantContext = Depends(get_tenant)` parameter to ALL four `/predict/*` handlers:
     ```python
     @fastapi_app.post("/predict/stuck", response_model=StuckResponse)
     async def predict_stuck(
         req: StuckRequest,
         tenant: TenantContext = Depends(get_tenant),
     ) -> StuckResponse:
         # tenant parameter available but not used yet (WP05 will use it)
         ...
     ```
  3. Apply to: `predict_stuck`, `predict_suggest`, `predict_duration`, `predict_quality`.
  4. Do NOT add tenant dependency to `/health`, `/status`, `/plugins`, or `/train` -- those are operational endpoints.
  5. The `tenant` parameter is available in handler bodies but not yet used. WP05 will add the model resolution logic that reads `tenant.tenant_id`.
- **Files**: `src/sigil_ml/routes.py` (modify)
- **Parallel?**: No -- depends on T014.
- **Notes**: Adding `Depends(get_tenant)` means FastAPI automatically calls the dependency before the handler body. In cloud mode, missing header triggers 401 BEFORE any prediction logic runs. In local mode, sentinel context is injected silently.

### Subtask T016 -- Verify 401 on missing tenant header (cloud) and sentinel in local mode

- **Purpose**: End-to-end verification that the two mode paths work correctly.
- **Steps**:
  1. Verify the 401 path: In cloud mode, a request to `/predict/stuck` without `X-Tenant-ID` header should return:
     ```json
     {"detail": "Missing required header 'X-Tenant-ID' for cloud mode."}
     ```
     with HTTP 401 status.
  2. Verify the local path: In local mode, a request to `/predict/stuck` without any header should succeed (the dependency injects `TenantContext(tenant_id="local", tier="local")`).
  3. Verify the happy path: In cloud mode, a request WITH `X-Tenant-ID: my-tenant` should succeed.
  4. Verify that `/health` and `/status` work WITHOUT any tenant header in both modes (they don't have the dependency).
  5. Add comments documenting these three behaviors if not already present.
- **Files**: `src/sigil_ml/tenant.py` (verify), `src/sigil_ml/routes.py` (verify)
- **Parallel?**: Yes -- verification task, can run alongside T017.

### Subtask T017 -- Make header name configurable via `SIGIL_TENANT_HEADER` env var

- **Purpose**: Allow operators to customize the tenant header name for API gateways that use different conventions.
- **Steps**:
  1. Verify that `tenant_header_name()` (from T014) reads `SIGIL_TENANT_HEADER`:
     ```python
     def tenant_header_name() -> str:
         return os.environ.get("SIGIL_TENANT_HEADER", "X-Tenant-ID")
     ```
  2. Verify `get_tenant_context()` calls `tenant_header_name()` dynamically (not hard-coded).
  3. Test: set `SIGIL_TENANT_HEADER=X-Custom-Tenant`, send request with that header, verify extraction works.
- **Files**: `src/sigil_ml/tenant.py` (verify)
- **Parallel?**: Yes -- independent verification.

## Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Circular imports | Low | Medium | Deferred import of `ServingMode` inside closure. `TYPE_CHECKING` guard for `AppState`. |
| Header injection by end users | Medium | Low | API gateway must strip user-provided tenant headers. Operational concern, documented in quickstart. |
| Performance overhead | Very Low | None | One header lookup + one dataclass construction per request. Negligible. |

## Review Guidance

- Verify cloud mode requests without `X-Tenant-ID` header return 401 for all `/predict/*` endpoints.
- Verify local mode requests without any tenant header succeed normally.
- Verify `/health` and `/status` work without a tenant header in both modes.
- Verify no circular imports between `tenant.py`, `config.py`, `app.py`, and `routes.py`.
- Check that `SIGIL_TENANT_HEADER` env var is respected by `tenant_header_name()`.
- Check that `TenantContext` is frozen (immutable).

## Activity Log

- 2026-03-30T01:45:14Z -- system -- lane=planned -- Prompt regenerated.
