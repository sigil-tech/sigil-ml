---
work_package_id: WP01
title: ModelStore Protocol & Local Implementation
lane: planned
dependencies: []
subtasks:
- T001
- T002
- T003
- T004
- T005
phase: Phase 1 - Foundation
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
- FR-001
- FR-002
- FR-003
- FR-004
---

# Work Package Prompt: WP01 -- ModelStore Protocol & Local Implementation

## Review Feedback

> **Populated by `/spec-kitty.review`** -- Reviewers add detailed feedback here when work needs changes.

*[This section is empty initially.]*

---

## Objectives & Success Criteria

- Define a `ModelStore` protocol (Python `Protocol` class) with `load()` and `save()` methods that all model classes will use for weight persistence.
- Implement `LocalModelStore` that preserves the exact current filesystem behavior: reading/writing `.joblib` files from `~/.local/share/sigild/ml-models/`.
- Implement a `model_store_factory()` function that selects the correct backend based on configuration.
- Add configuration entries to `config.py` for backend selection.
- After this WP, the abstraction layer is in place and local behavior is functionally identical to today.

**Success Criteria**:
- `ModelStore` protocol is importable from `sigil_ml.storage`.
- `LocalModelStore` can save and load bytes, producing files at the exact same paths as current `config.weights_path()`.
- `model_store_factory()` returns `LocalModelStore` when mode is `local` (default).
- Missing model files return `None` (not an exception).
- Corrupted files return `None` with a warning log.

## Context & Constraints

- **Spec**: `kitty-specs/003-model-storage-abstraction/spec.md` -- FR-001 through FR-004.
- **Current code**: All 5 model classes in `src/sigil_ml/models/` currently call `config.weights_path(name)` and use `joblib.load(path)` / `joblib.dump(obj, path)` directly. `QualityEstimator` uses `json.load`/`json.dump` with `open()`.
- **Current config**: `src/sigil_ml/config.py` provides `models_dir()` and `weights_path(model_name)`.
- **Constraint**: No heavyweight dependencies. The protocol uses Python's `typing.Protocol` (stdlib).
- **Constraint**: The protocol must be synchronous. All current code is sync; async can be added later.
- **Constraint**: `LocalModelStore` must create the models directory if it doesn't exist (matching current `models_dir()` behavior).

## Subtasks & Detailed Guidance

### Subtask T001 -- Define ModelStore protocol in `src/sigil_ml/storage/__init__.py`

- **Purpose**: Establish the contract that all storage backends must fulfill. This is the cornerstone interface that decouples model classes from storage implementation.

- **Steps**:
  1. Create directory `src/sigil_ml/storage/` with `__init__.py`.
  2. Define a `Protocol` class named `ModelStore`:
     ```python
     from typing import Protocol

     class ModelStore(Protocol):
         def load(self, model_name: str) -> bytes | None:
             """Load serialized model weights by name.

             Returns the raw bytes of the model file, or None if the model
             does not exist or cannot be read.
             """
             ...

         def save(self, model_name: str, data: bytes) -> None:
             """Save serialized model weights by name.

             Args:
                 model_name: Identifier like "stuck", "activity", "workflow",
                             "duration", "quality".
                 data: Raw bytes of the serialized model (joblib or JSON).
             """
             ...
     ```
  3. Export `ModelStore` from the package `__init__.py`.

- **Files**:
  - Create: `src/sigil_ml/storage/__init__.py`

- **Parallel?**: No -- other subtasks depend on this definition.

- **Notes**:
  - Use `from __future__ import annotations` for forward-reference compatibility.
  - The protocol intentionally does not include `delete()` or `list()` -- those can be added later if needed.
  - Model names are simple strings: `"stuck"`, `"activity"`, `"workflow"`, `"duration"`, `"quality"` -- matching the names used in `config.weights_path()` today.

### Subtask T002 -- Implement LocalModelStore in `src/sigil_ml/storage/local.py`

- **Purpose**: Preserve the exact current local filesystem behavior behind the `ModelStore` interface. This is the default backend and must produce identical files at identical paths.

- **Steps**:
  1. Create `src/sigil_ml/storage/local.py`.
  2. Implement `LocalModelStore`:
     ```python
     import logging
     from pathlib import Path

     logger = logging.getLogger(__name__)

     class LocalModelStore:
         """ModelStore backed by the local filesystem.

         Reads and writes .joblib files from a configured directory.
         Default: ~/.local/share/sigild/ml-models/
         """

         def __init__(self, models_dir: Path) -> None:
             self._dir = models_dir
             self._dir.mkdir(parents=True, exist_ok=True)

         def load(self, model_name: str) -> bytes | None:
             path = self._dir / f"{model_name}.joblib"
             if not path.exists():
                 return None
             try:
                 return path.read_bytes()
             except Exception:
                 logger.warning("Failed to read model file: %s", path)
                 return None

         def save(self, model_name: str, data: bytes) -> None:
             path = self._dir / f"{model_name}.joblib"
             self._dir.mkdir(parents=True, exist_ok=True)
             path.write_bytes(data)
             logger.info("Saved model to %s", path)
     ```
  3. The constructor takes `models_dir: Path` rather than reading config directly -- this keeps the class testable and config-free.

- **Files**:
  - Create: `src/sigil_ml/storage/local.py`

- **Parallel?**: No -- depends on T001 for the protocol definition (though it doesn't need to `import` it; it just needs to satisfy the protocol structurally).

- **Notes**:
  - `load()` returns raw bytes, not deserialized objects. Deserialization (`joblib.load(BytesIO(data))`) is the model class's responsibility.
  - `save()` accepts raw bytes. Serialization (`joblib.dump(model, BytesIO())`) is the model class's responsibility.
  - The `.joblib` extension is hardcoded to match current convention. `QualityEstimator` currently saves as `.joblib` too (the file contains JSON but uses the `.joblib` extension via `config.weights_path("quality")`).
  - Directory creation in `__init__` and `save()` matches current `models_dir()` behavior which calls `mkdir(parents=True, exist_ok=True)`.

### Subtask T003 -- Add model_store_factory() function in `src/sigil_ml/storage/factory.py`

- **Purpose**: Centralize backend selection logic so that `app.py` and other entry points can get the right `ModelStore` with a single call.

- **Steps**:
  1. Create `src/sigil_ml/storage/factory.py`.
  2. Implement the factory:
     ```python
     from sigil_ml import config
     from sigil_ml.storage.local import LocalModelStore

     def model_store_factory(
         backend: str | None = None,
         tenant_id: str | None = None,
     ) -> "ModelStore":
         """Create a ModelStore instance based on configuration.

         Args:
             backend: Override backend selection. Values: "local", "s3".
                      If None, reads from config.
             tenant_id: Tenant ID for S3 prefix (cloud mode only).

         Returns:
             A ModelStore implementation.
         """
         backend = backend or config.model_store_backend()

         if backend == "local":
             return LocalModelStore(config.models_dir())

         if backend == "s3":
             # S3 import deferred to avoid requiring boto3 in local mode
             raise NotImplementedError(
                 "S3 backend not yet implemented. "
                 "Install sigil-ml[cloud] and see WP02."
             )

         raise ValueError(f"Unknown model store backend: {backend!r}")
     ```
  3. Export from `src/sigil_ml/storage/__init__.py`.

- **Files**:
  - Create: `src/sigil_ml/storage/factory.py`
  - Update: `src/sigil_ml/storage/__init__.py` (add export)

- **Parallel?**: No -- depends on T001 and T002.

- **Notes**:
  - The `NotImplementedError` for S3 is intentional -- WP02 fills it in. This keeps WP01 self-contained.
  - The factory is a plain function, not a class. It can be replaced with a more sophisticated builder later.
  - `tenant_id` parameter is accepted but only used by S3 backend. Local mode ignores it.

### Subtask T004 -- Add configuration entries for model storage backend in `src/sigil_ml/config.py`

- **Purpose**: Provide config-driven backend selection so the factory knows which `ModelStore` to instantiate.

- **Steps**:
  1. Add to `src/sigil_ml/config.py`:
     ```python
     def serving_mode() -> str:
         """Return the serving mode: 'local' (default) or 'cloud'."""
         return os.environ.get("SIGIL_MODE", "local").lower()

     def model_store_backend() -> str:
         """Return the model storage backend: 'local' or 's3'.

         Defaults based on serving mode:
         - local mode -> 'local'
         - cloud mode -> 's3'

         Override with SIGIL_MODEL_STORE env var.
         """
         explicit = os.environ.get("SIGIL_MODEL_STORE", "").lower()
         if explicit in ("local", "s3"):
             return explicit
         # Infer from serving mode
         mode = serving_mode()
         return "s3" if mode == "cloud" else "local"

     def model_cache_ttl() -> int:
         """Return model cache TTL in seconds (default 300 = 5 minutes)."""
         try:
             return int(os.environ.get("SIGIL_MODEL_CACHE_TTL", "300"))
         except ValueError:
             return 300

     def s3_bucket() -> str:
         """Return the S3 bucket name for model storage."""
         return os.environ.get("SIGIL_MODEL_BUCKET", "")

     def s3_region() -> str:
         """Return the S3 region."""
         return os.environ.get("SIGIL_MODEL_REGION", "us-east-1")

     def s3_endpoint_url() -> str | None:
         """Return optional S3 endpoint URL (for MinIO or S3-compatible stores)."""
         url = os.environ.get("SIGIL_MODEL_ENDPOINT_URL", "")
         return url if url else None
     ```

- **Files**:
  - Update: `src/sigil_ml/config.py`

- **Parallel?**: Yes -- can proceed alongside T001-T002 since it only adds new functions to config.

- **Notes**:
  - `serving_mode()` is the canonical way to check local vs cloud. Feature 001 (Cloud Serving Mode) will also use this.
  - S3-specific config functions (`s3_bucket`, `s3_region`, `s3_endpoint_url`) are defined here but only called by `S3ModelStore` in WP02. Defining them early keeps config centralized.
  - Environment variable names use the `SIGIL_` prefix for consistency with the project.

### Subtask T005 -- Verify LocalModelStore handles edge cases

- **Purpose**: Ensure the `LocalModelStore` correctly handles all edge cases specified in the spec: missing files, corrupted files, and directory creation.

- **Steps**:
  1. Verify `load()` returns `None` when the model file does not exist (no exception raised).
  2. Verify `load()` returns `None` and logs a warning when the file exists but is unreadable (e.g., permission error simulation).
  3. Verify `save()` creates the models directory if it doesn't exist.
  4. Verify `save()` overwrites an existing file without error.
  5. Verify `load()` returns the exact bytes that were passed to `save()` (round-trip integrity).

- **Files**:
  - Verify behavior in: `src/sigil_ml/storage/local.py`

- **Parallel?**: No -- requires T002 to be complete.

- **Notes**:
  - This is a manual verification subtask during implementation. If tests are later requested, these scenarios become test cases.
  - The round-trip check is critical: `save("stuck", data)` followed by `load("stuck")` must return exactly `data`.

## Risks & Mitigations

- **Risk**: `Protocol` class might not be recognized by type checkers for older Python versions. **Mitigation**: `requires-python = ">=3.10"` in `pyproject.toml` ensures `Protocol` is fully supported.
- **Risk**: Creating a new `storage/` package might conflict with other planned packages. **Mitigation**: The package name `sigil_ml.storage` is specific and unlikely to conflict. Feature 002 (DataStore) uses `sigil_ml.data_store` or similar.

## Review Guidance

- Verify the `ModelStore` protocol has exactly `load` and `save` methods with the correct signatures.
- Verify `LocalModelStore` produces files at the same paths as current `config.weights_path()`.
- Verify the factory function defaults to `local` backend.
- Verify all new config functions have sensible defaults and read from environment variables.
- Verify no existing code is broken -- this WP only adds new files and functions.

## Implementation Command

```bash
spec-kitty implement WP01
```

## Activity Log

- 2026-03-29T16:30:00Z -- system -- lane=planned -- Prompt created.
