---
work_package_id: WP02
title: S3ModelStore Implementation
lane: planned
dependencies: [WP01]
subtasks:
- T006
- T007
- T008
- T009
- T010
phase: Phase 2 - Cloud Storage
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
- FR-005
- FR-006
---

# Work Package Prompt: WP02 -- S3ModelStore Implementation

## Review Feedback

> **Populated by `/spec-kitty.review`** -- Reviewers add detailed feedback here when work needs changes.

*[This section is empty initially.]*

---

## Objectives & Success Criteria

- Implement `S3ModelStore` that reads and writes model weight files from an S3-compatible object store.
- Support per-tenant key prefixes so that each tenant's models are isolated in the bucket.
- Handle all S3 error conditions gracefully: missing bucket, invalid credentials, network errors, missing keys, corrupted objects.
- Add `boto3` as an optional dependency under a `cloud` extras group.
- Update the `model_store_factory()` to wire S3 backend when configured.

**Success Criteria**:
- `S3ModelStore.save("stuck", data)` writes to `s3://{bucket}/{tenant_id}/stuck.joblib`.
- `S3ModelStore.load("stuck")` returns the bytes from that S3 key, or `None` if the key doesn't exist.
- Missing keys return `None` without raising exceptions.
- Invalid credentials or missing bucket cause a clear error at construction time (fail-fast).
- `pip install sigil-ml[cloud]` installs `boto3`.

## Context & Constraints

- **Spec**: `kitty-specs/003-model-storage-abstraction/spec.md` -- FR-005, FR-006.
- **Depends on WP01**: `ModelStore` protocol defined in `src/sigil_ml/storage/__init__.py`, factory in `src/sigil_ml/storage/factory.py`, config functions in `src/sigil_ml/config.py`.
- **Constraint**: `boto3` must be optional -- local-mode users must not need it installed. Use a try/except import guard.
- **Constraint**: No heavyweight dependencies beyond `boto3`. Do not add `aiobotocore` or `s3fs`.
- **Constraint**: Synchronous API. `boto3`'s default client is synchronous, which matches the `ModelStore` protocol.
- **Edge case (from spec)**: Concurrent model saves use S3's last-writer-wins semantics. No locking required.
- **Edge case (from spec)**: S3 temporary unreachability should cause `load()` to return `None` (serving falls back to rule-based). The system retries on subsequent requests naturally.

## Subtasks & Detailed Guidance

### Subtask T006 -- Implement S3ModelStore in `src/sigil_ml/storage/s3.py`

- **Purpose**: Provide the S3-backed storage backend that enables cloud deployments to load and save model weights from object storage with per-tenant isolation.

- **Steps**:
  1. Create `src/sigil_ml/storage/s3.py`.
  2. Implement `S3ModelStore`:
     ```python
     from __future__ import annotations

     import logging
     from typing import TYPE_CHECKING

     logger = logging.getLogger(__name__)

     try:
         import boto3
         from botocore.exceptions import ClientError, NoCredentialsError
         _HAS_BOTO3 = True
     except ImportError:
         _HAS_BOTO3 = False

     class S3ModelStore:
         """ModelStore backed by S3-compatible object storage.

         Key format: {prefix}/{model_name}.joblib
         Where prefix is typically the tenant ID.
         """

         def __init__(
             self,
             bucket: str,
             prefix: str = "",
             region: str = "us-east-1",
             endpoint_url: str | None = None,
         ) -> None:
             if not _HAS_BOTO3:
                 raise ImportError(
                     "boto3 is required for S3 model storage. "
                     "Install with: pip install sigil-ml[cloud]"
                 )

             self._bucket = bucket
             self._prefix = prefix.strip("/")
             self._client = boto3.client(
                 "s3",
                 region_name=region,
                 endpoint_url=endpoint_url,
             )

             # Validate bucket access at construction time
             self._validate_bucket()

         def _validate_bucket(self) -> None:
             """Verify the bucket exists and is accessible."""
             try:
                 self._client.head_bucket(Bucket=self._bucket)
                 logger.info(
                     "S3ModelStore: connected to bucket %s (prefix=%s)",
                     self._bucket, self._prefix,
                 )
             except ClientError as e:
                 code = e.response["Error"]["Code"]
                 if code == "404":
                     raise ValueError(
                         f"S3 bucket '{self._bucket}' does not exist. "
                         f"Create it or check SIGIL_MODEL_BUCKET."
                     ) from e
                 if code == "403":
                     raise PermissionError(
                         f"Access denied to S3 bucket '{self._bucket}'. "
                         f"Check IAM permissions."
                     ) from e
                 raise
             except NoCredentialsError as e:
                 raise PermissionError(
                     "No AWS credentials found. Configure AWS_ACCESS_KEY_ID "
                     "and AWS_SECRET_ACCESS_KEY, or use an IAM role."
                 ) from e

         def _key(self, model_name: str) -> str:
             """Build the S3 object key for a model."""
             if self._prefix:
                 return f"{self._prefix}/{model_name}.joblib"
             return f"{model_name}.joblib"

         def load(self, model_name: str) -> bytes | None:
             key = self._key(model_name)
             try:
                 response = self._client.get_object(
                     Bucket=self._bucket, Key=key
                 )
                 data = response["Body"].read()
                 logger.info(
                     "S3ModelStore: loaded %s (%d bytes)", key, len(data)
                 )
                 return data
             except ClientError as e:
                 code = e.response["Error"]["Code"]
                 if code == "NoSuchKey":
                     logger.debug(
                         "S3ModelStore: model not found: %s", key
                     )
                     return None
                 logger.warning(
                     "S3ModelStore: error loading %s: %s", key, e
                 )
                 return None
             except Exception:
                 logger.warning(
                     "S3ModelStore: unexpected error loading %s",
                     key, exc_info=True,
                 )
                 return None

         def save(self, model_name: str, data: bytes) -> None:
             key = self._key(model_name)
             try:
                 self._client.put_object(
                     Bucket=self._bucket,
                     Key=key,
                     Body=data,
                 )
                 logger.info(
                     "S3ModelStore: saved %s (%d bytes)", key, len(data)
                 )
             except Exception:
                 logger.exception(
                     "S3ModelStore: failed to save %s", key
                 )
                 raise
     ```

- **Files**:
  - Create: `src/sigil_ml/storage/s3.py`

- **Parallel?**: No -- core implementation.

- **Notes**:
  - The `_HAS_BOTO3` guard ensures a clear error message when boto3 is not installed, rather than a confusing `ImportError` deep in the stack.
  - `_validate_bucket()` is called at construction time (fail-fast). This ensures misconfiguration is caught at startup, not on the first prediction request.
  - `load()` catches all exceptions and returns `None` -- this aligns with the spec requirement that the system falls back to rule-based predictions when the backend is unreachable.
  - `save()` re-raises exceptions because a save failure during training should be surfaced to the caller (training can retry or log).
  - The `prefix` field is the tenant ID. Key format: `{tenant_id}/stuck.joblib`.

### Subtask T007 -- Add S3 configuration env vars to `src/sigil_ml/config.py`

- **Purpose**: The S3 config functions were defined in WP01/T004 but may need refinement. Verify they are complete and add any missing entries.

- **Steps**:
  1. Verify `config.py` has: `s3_bucket()`, `s3_region()`, `s3_endpoint_url()`.
  2. Add `tenant_id()` if not present:
     ```python
     def tenant_id() -> str:
         """Return the tenant ID for multi-tenant model isolation.

         In cloud mode, this determines the S3 key prefix.
         Read from SIGIL_TENANT_ID env var or X-Tenant-ID header (per-request).
         This function returns the default/startup tenant ID.
         """
         return os.environ.get("SIGIL_TENANT_ID", "default")
     ```
  3. Verify all env vars are documented in code comments.

- **Files**:
  - Update: `src/sigil_ml/config.py`

- **Parallel?**: No -- T006 references these config values.

- **Notes**:
  - Per-request tenant ID (from HTTP headers) is handled at the route level, not in config. The config `tenant_id()` is the startup default.
  - `SIGIL_MODEL_ENDPOINT_URL` supports MinIO and other S3-compatible stores for local development.

### Subtask T008 -- Handle S3 error conditions

- **Purpose**: Ensure all S3 failure modes are handled gracefully per the spec's edge cases section.

- **Steps**:
  1. **Missing bucket**: `_validate_bucket()` raises `ValueError` with actionable message (already in T006).
  2. **Invalid credentials**: `_validate_bucket()` raises `PermissionError` with clear message (already in T006).
  3. **Network timeout**: `load()` catches all exceptions, logs warning, returns `None`. The service falls back to rule-based predictions. Verify `boto3` client has a reasonable timeout config:
     ```python
     from botocore.config import Config
     self._client = boto3.client(
         "s3",
         region_name=region,
         endpoint_url=endpoint_url,
         config=Config(
             connect_timeout=5,
             read_timeout=10,
             retries={"max_attempts": 2},
         ),
     )
     ```
  4. **Corrupted object**: The raw bytes are returned by `load()`. The model class that deserializes with `joblib.load(BytesIO(data))` will catch the deserialization error and fall back. The store layer does not need to validate content.
  5. **Concurrent saves**: S3 last-writer-wins. No locking needed. Verify `save()` does not use conditional writes.

- **Files**:
  - Update: `src/sigil_ml/storage/s3.py`

- **Parallel?**: Yes -- can be developed alongside T006 as error handling refinements.

- **Notes**:
  - The `Config(connect_timeout=5, read_timeout=10, retries={"max_attempts": 2})` keeps requests from hanging indefinitely.
  - If the network is down, `load()` returns `None` after ~15 seconds (5s connect + 10s read, 2 retries). The serving layer falls back to rule-based predictions.

### Subtask T009 -- Update model_store_factory() for S3 backend

- **Purpose**: Replace the `NotImplementedError` in the factory with actual S3 backend instantiation.

- **Steps**:
  1. Update `src/sigil_ml/storage/factory.py`:
     ```python
     if backend == "s3":
         from sigil_ml.storage.s3 import S3ModelStore
         return S3ModelStore(
             bucket=config.s3_bucket(),
             prefix=tenant_id or config.tenant_id(),
             region=config.s3_region(),
             endpoint_url=config.s3_endpoint_url(),
         )
     ```
  2. The S3 import is deferred (inside the `if` block) so that `boto3` is only imported when actually needed.

- **Files**:
  - Update: `src/sigil_ml/storage/factory.py`

- **Parallel?**: No -- depends on T006.

- **Notes**:
  - `tenant_id` parameter takes precedence over config default. This supports per-request tenant switching in cloud mode.

### Subtask T010 -- Add boto3 as optional dependency in pyproject.toml

- **Purpose**: Allow cloud users to install S3 support without burdening local-mode users.

- **Steps**:
  1. Update `pyproject.toml` to add a `cloud` extras group:
     ```toml
     [project.optional-dependencies]
     dev = ["pytest>=8.0", "httpx>=0.27", "ruff>=0.4", "pyre-check>=0.9.18"]
     cloud = ["boto3>=1.34"]
     ```
  2. Verify that `pip install sigil-ml` (without extras) does NOT install `boto3`.
  3. Verify that `pip install sigil-ml[cloud]` installs `boto3` and its dependencies.

- **Files**:
  - Update: `pyproject.toml`

- **Parallel?**: Yes -- independent of the implementation code.

- **Notes**:
  - `boto3>=1.34` is a recent stable release. Pin the minimum version to avoid known issues with older releases.
  - Do NOT add `botocore` separately -- it is a dependency of `boto3` and will be installed automatically.

## Risks & Mitigations

- **Risk**: `boto3` dependency size (~100MB installed). **Mitigation**: Optional extras group. Local users never see it.
- **Risk**: AWS credential management complexity. **Mitigation**: `boto3` supports standard credential chain (env vars, IAM role, shared credentials file). No custom credential management needed.
- **Risk**: S3 eventual consistency for PUT-then-GET. **Mitigation**: S3 now provides strong read-after-write consistency for PUTs. This is a non-issue for modern S3.
- **Risk**: MinIO compatibility differences. **Mitigation**: MinIO implements the S3 API. The `endpoint_url` parameter enables pointing to any S3-compatible store.

## Review Guidance

- Verify `S3ModelStore` satisfies the `ModelStore` protocol (same `load`/`save` signatures).
- Verify bucket validation happens at construction time with clear error messages.
- Verify `load()` never raises exceptions -- always returns `None` on failure.
- Verify `save()` raises on failure (so training pipeline is aware).
- Verify `boto3` import is guarded with try/except.
- Verify `pyproject.toml` has `cloud` extras group.

## Implementation Command

```bash
spec-kitty implement WP02 --base WP01
```

## Activity Log

- 2026-03-29T16:30:00Z -- system -- lane=planned -- Prompt created.
