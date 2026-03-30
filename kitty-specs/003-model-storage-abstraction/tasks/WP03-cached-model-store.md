---
work_package_id: "WP03"
title: "CachedModelStore with TTL"
lane: "planned"
dependencies: ["WP01"]
subtasks:
  - "T012"
  - "T013"
  - "T014"
  - "T015"
  - "T016"
phase: "Phase 2 - Cloud Storage"
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
  - "FR-008"
  - "FR-009"
---

# Work Package Prompt: WP03 -- CachedModelStore with TTL

## Review Feedback

> **Populated by `/spec-kitty.review`** -- Reviewers add detailed feedback here when work needs changes.

*[This section is empty initially.]*

---

## Markdown Formatting
Wrap HTML/XML tags in backticks: `` `<div>` ``, `` `<script>` ``
Use language identifiers in code blocks: ````python`, ````bash`

---

## Objectives & Success Criteria

- Implement `CachedModelStore` as a decorator that wraps any `ModelStore` implementation with in-memory TTL-based caching.
- The cache stores raw bytes keyed by `model_name` with configurable TTL.
- Thread-safety for concurrent HTTP request handling in cloud mode.
- Cache eviction on error and manual clearing for testing.
- The `CachedModelStore` itself satisfies the `ModelStore` protocol (drop-in replacement).

**Success Criteria**:
- First `load("stuck")` calls the inner store; second call within TTL returns cached bytes without calling inner store.
- After TTL expires, next `load()` delegates to inner store again and updates the cache.
- `save()` writes through to inner store AND updates the cache (no stale reads after save).
- Concurrent `load()` calls do not cause data corruption.
- `evict()` removes a specific entry; `clear()` removes all entries.
- Cached model load adds <1ms overhead (SC-004 from spec).

## Context & Constraints

- **Spec**: `kitty-specs/003-model-storage-abstraction/spec.md` -- FR-008, FR-009.
- **Plan**: Design decision D3 (CachedModelStore as decorator), component design section.
- **Depends on WP01**: `ModelStore` protocol in `src/sigil_ml/storage/model_store.py`.
- **Constraint**: No external caching libraries. Use stdlib `dict` + `threading.Lock`.
- **Constraint**: Default TTL is 300 seconds (5 minutes), configurable via `SIGIL_MODEL_CACHE_TTL` (already in `config.py` from WP01).
- **Constraint**: Cache stores raw bytes, not deserialized model objects. Deserialization is the model class's responsibility. This keeps the cache layer simple and model-agnostic.
- **Constraint**: Per plan.md D4, tenant isolation is handled by having one `S3ModelStore` per tenant. The cache wraps a single tenant's store, so `model_name` alone is a sufficient cache key.

## Subtasks & Detailed Guidance

### Subtask T012 -- Implement CachedModelStore in `src/sigil_ml/storage/model_store.py`

- **Purpose**: Provide in-memory caching around any `ModelStore` to eliminate repeated remote reads in cloud mode. Without this, every prediction request in cloud mode would require an S3 round-trip (~50-200ms), which is unacceptable for interactive use.

- **Steps**:
  1. Add `CachedModelStore` to `src/sigil_ml/storage/model_store.py` (same file as protocol and other implementations):
     ```python
     import threading
     import time


     class CachedModelStore:
         """In-memory cache wrapping a ModelStore with TTL-based expiration.

         Satisfies the ModelStore protocol. Used in cloud mode to avoid
         repeated S3 reads for model weights.

         The cache stores raw bytes keyed by model_name. Tenant isolation
         is handled by wrapping a per-tenant S3ModelStore instance.
         """

         def __init__(self, inner: ModelStore, ttl_seconds: float = 300.0) -> None:
             self._inner = inner
             self._ttl = ttl_seconds
             self._cache: dict[str, tuple[bytes, float]] = {}
             self._lock = threading.Lock()

         def load(self, model_name: str) -> bytes | None:
             now = time.time()

             # Check cache under lock
             with self._lock:
                 entry = self._cache.get(model_name)
                 if entry is not None:
                     data, loaded_at = entry
                     if (now - loaded_at) < self._ttl:
                         logger.debug("CachedModelStore: cache hit for %s", model_name)
                         return data

             # Cache miss or expired -- fetch from inner store (outside lock)
             logger.debug("CachedModelStore: cache miss for %s", model_name)
             data = self._inner.load(model_name)

             if data is not None:
                 with self._lock:
                     self._cache[model_name] = (data, time.time())

             return data

         def save(self, model_name: str, data: bytes) -> None:
             # Write through to inner store first
             self._inner.save(model_name, data)

             # Update cache with fresh data
             with self._lock:
                 self._cache[model_name] = (data, time.time())
             logger.debug("CachedModelStore: cache updated after save for %s", model_name)

         def exists(self, model_name: str) -> bool:
             with self._lock:
                 if model_name in self._cache:
                     return True
             return self._inner.exists(model_name)

         def evict(self, model_name: str) -> None:
             """Remove a specific entry from the cache.

             Call when deserialization of cached data fails, so the
             next load() re-fetches from the inner store.
             """
             with self._lock:
                 self._cache.pop(model_name, None)
             logger.info("CachedModelStore: evicted %s", model_name)

         def clear(self) -> None:
             """Remove all entries from the cache."""
             with self._lock:
                 self._cache.clear()
             logger.info("CachedModelStore: cache cleared")
     ```

- **Files**:
  - Update: `src/sigil_ml/storage/model_store.py`
  - Update: `src/sigil_ml/storage/__init__.py` (add `CachedModelStore` to imports/exports)

- **Parallel?**: No -- core implementation.

- **Notes**:
  - Cache entries are `tuple[bytes, float]` -- the bytes and the timestamp when they were loaded.
  - `evict()` and `clear()` are extra methods not on the `ModelStore` protocol. They are used for error recovery and testing. Callers check with `hasattr(store, 'evict')` before calling.
  - The import for `threading` and `time` should be at the top of `model_store.py` (module-level).

### Subtask T013 -- Implement TTL-based expiration

- **Purpose**: Ensure cached entries are refreshed after TTL expires, so updated models from training are picked up within the configured time window.

- **Steps**:
  1. The TTL check is in `load()`:
     ```python
     if (now - loaded_at) < self._ttl:
         return data  # Cache hit
     # else: cache miss, fall through to inner.load()
     ```
  2. On cache miss/expiry, `inner.load()` fetches fresh data. If fresh data is returned, it replaces the stale entry.
  3. If `inner.load()` returns `None` (model deleted or error), the stale cache entry remains until its next access. **Decision**: Do NOT cache `None` values to avoid caching transient errors. If the inner store returns `None`, the stale entry is effectively evicted on next access because the TTL check already failed.
  4. Default TTL is `config.model_cache_ttl()` = 300 seconds. Constructor accepts override for testing.

- **Files**:
  - Verify: `src/sigil_ml/storage/model_store.py`

- **Parallel?**: No -- part of T012 implementation.

- **Notes**:
  - The spec states cache TTL of 5 minutes (FR-008). This is the default.
  - Operators can tune via `SIGIL_MODEL_CACHE_TTL` env var for faster model pickup (lower TTL) or reduced S3 traffic (higher TTL).
  - There is no background eviction thread. Stale entries are only replaced when accessed. This is simpler and avoids threading complexity.

### Subtask T014 -- Implement write-through save

- **Purpose**: When `save()` is called (after training), the cache must be updated immediately so that subsequent `load()` calls return the fresh data without waiting for TTL expiry.

- **Steps**:
  1. `save()` delegates to `inner.save()` first (the actual persistence).
  2. If `inner.save()` succeeds (no exception), update the cache:
     ```python
     with self._lock:
         self._cache[model_name] = (data, time.time())
     ```
  3. If `inner.save()` raises an exception, do NOT update the cache -- the save failed and old data is still current.
  4. The fresh timestamp means the TTL clock resets on save.

- **Files**:
  - Verify: `src/sigil_ml/storage/model_store.py`

- **Parallel?**: No -- part of T012 implementation.

- **Notes**:
  - Write-through ensures the same process that trains immediately sees the new model. Other processes/pods will pick up the change when their cache TTL expires.
  - This matches spec User Story 3 acceptance scenario 3: after training saves updated weights, the cache reflects them immediately in the same instance.

### Subtask T015 -- Add thread-safety with Lock

- **Purpose**: Cloud mode serves concurrent HTTP requests via uvicorn workers. Cache operations must be thread-safe to prevent data corruption in the shared `_cache` dict.

- **Steps**:
  1. Use a single `threading.Lock` for all cache dict operations.
  2. **Critical pattern**: The lock is RELEASED before calling `self._inner.load()`:
     ```python
     def load(self, model_name: str) -> bytes | None:
         now = time.time()

         # Check cache under lock (fast: dict lookup)
         with self._lock:
             entry = self._cache.get(model_name)
             if entry is not None:
                 data, loaded_at = entry
                 if (now - loaded_at) < self._ttl:
                     return data  # Hit

         # Fetch without lock (slow: S3 network call)
         data = self._inner.load(model_name)

         # Update cache under lock (fast: dict write)
         if data is not None:
             with self._lock:
                 self._cache[model_name] = (data, time.time())

         return data
     ```
  3. **Trade-off**: Two threads hitting the same expired key may both fetch from S3. This is acceptable:
     - The fetch is idempotent (both get the same bytes).
     - The last write to the cache wins (both write the same data).
     - This avoids blocking all model loads while one fetches from S3.
  4. The lock is held only for brief dict operations (microseconds). S3 fetches happen outside the lock.

- **Files**:
  - Verify: `src/sigil_ml/storage/model_store.py`

- **Parallel?**: Yes -- can be developed alongside T012-T014 as a refinement.

- **Notes**:
  - Python's GIL provides some protection for dict operations, but explicit locking is needed for the check-then-act pattern (check TTL, then fetch, then update).
  - A per-key lock could be added later if the global lock proves to be a bottleneck. For typical workloads (5 models, <1 second fetch), the global lock is sufficient.
  - `save()` and `evict()` also acquire the lock for their dict operations.

### Subtask T016 -- Add evict() and clear() methods

- **Purpose**: Provide explicit cache invalidation for error recovery (corrupted model data) and for testing.

- **Steps**:
  1. `evict(model_name)`: Remove a single entry by key. Used by model classes when `joblib.load(BytesIO(data))` fails:
     ```python
     # In model class (WP04):
     data = self._store.load("stuck")
     if data is not None:
         try:
             self.model = joblib.load(BytesIO(data))
         except Exception:
             if hasattr(self._store, 'evict'):
                 self._store.evict("stuck")
             self.model = None
     ```
  2. `clear()`: Remove all entries. Useful for testing and for forced cache refresh.
  3. Both methods acquire the lock before modifying the dict.
  4. Both methods log at INFO level for audit trail.

- **Files**:
  - Verify: `src/sigil_ml/storage/model_store.py`

- **Parallel?**: No -- depends on T012 cache structure.

- **Notes**:
  - `evict()` and `clear()` are NOT on the `ModelStore` protocol. They are specific to `CachedModelStore`. Callers use `hasattr(store, 'evict')` to check availability.
  - After `evict()`, the next `load()` for that key will delegate to the inner store, re-fetching from S3 (or wherever the inner store reads from).
  - `evict()` with a key that doesn't exist in the cache is a no-op (uses `dict.pop(key, None)`).

## Risks & Mitigations

- **Risk**: Memory growth from caching many models. **Mitigation**: ~5 models per tenant, each typically <10MB. With 300s TTL, stale entries are replaced naturally. A max-entry eviction policy can be added later if needed.
- **Risk**: Thundering herd on cache expiry. **Mitigation**: The lock-release-before-fetch pattern means only the requesting threads fetch, and they don't block others. Multiple fetches for the same key are idempotent.
- **Risk**: Cache returns stale data after training. **Mitigation**: `save()` is write-through, updating the cache immediately. Other instances pick up changes within TTL.
- **Risk**: Cached `None` values mask transient errors. **Mitigation**: `None` is NOT cached. If the inner store returns `None` due to a transient error, the next `load()` retries.

## Review Guidance

- Verify `CachedModelStore` satisfies the `ModelStore` protocol (has `load`, `save`, `exists` with correct signatures).
- Verify TTL logic: check `time.time() - loaded_at < self._ttl` inside the lock.
- Verify `save()` is write-through: inner store called first, cache updated on success.
- Verify the lock is NOT held during `self._inner.load()` calls.
- Verify `evict()` and `clear()` methods exist and are thread-safe.
- Verify `None` results from inner store are NOT cached.
- Verify `CachedModelStore` is exported from `src/sigil_ml/storage/__init__.py`.

## Implementation Command

```bash
spec-kitty implement WP03 --base WP01
```

## Activity Log

- 2026-03-30T01:45:11Z -- system -- lane=planned -- Prompt created.
