---
work_package_id: WP03
title: ModelCache with TTL
lane: planned
dependencies: [WP01]
subtasks:
- T011
- T012
- T013
- T014
- T015
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
- FR-008
- FR-009
---

# Work Package Prompt: WP03 -- ModelCache with TTL

## Review Feedback

> **Populated by `/spec-kitty.review`** -- Reviewers add detailed feedback here when work needs changes.

*[This section is empty initially.]*

---

## Objectives & Success Criteria

- Implement an in-memory caching layer (`ModelCache`) that wraps any `ModelStore` implementation.
- Cache loaded model bytes in memory with a configurable TTL, keyed by tenant ID and model name.
- Ensure thread-safety for concurrent request handling in cloud mode.
- Handle cache eviction on errors (corrupted or stale entries).

**Success Criteria**:
- `ModelCache` wrapping a mock store: first `load("stuck")` calls the inner store; second call within TTL returns cached bytes without calling the inner store.
- After TTL expires, the next `load()` delegates to the inner store again.
- `save()` updates the cache and delegates to the inner store.
- Concurrent `load()` calls for the same model do not cause race conditions.
- Cache key format supports multi-tenant isolation: different tenants have independent cache entries.

## Context & Constraints

- **Spec**: `kitty-specs/003-model-storage-abstraction/spec.md` -- FR-008, FR-009.
- **Depends on WP01**: `ModelStore` protocol definition.
- **Constraint**: No external caching libraries. Use stdlib `dict` + `threading.Lock`.
- **Constraint**: Default TTL is 300 seconds (5 minutes), configurable via `SIGIL_MODEL_CACHE_TTL` env var (already defined in WP01/T004 config).
- **Constraint**: The `ModelCache` must itself satisfy the `ModelStore` protocol so it can be used as a drop-in replacement.
- **Design decision**: The cache stores raw bytes, not deserialized model objects. Deserialization is the model class's responsibility. This keeps the cache layer simple and model-agnostic.

## Subtasks & Detailed Guidance

### Subtask T011 -- Implement ModelCache in `src/sigil_ml/storage/cache.py`

- **Purpose**: Provide an in-memory caching decorator around any `ModelStore` to avoid repeated remote reads in cloud mode. This is critical for acceptable prediction latency.

- **Steps**:
  1. Create `src/sigil_ml/storage/cache.py`.
  2. Implement `ModelCache`:
     ```python
     from __future__ import annotations

     import logging
     import threading
     import time
     from dataclasses import dataclass, field
     from typing import TYPE_CHECKING

     if TYPE_CHECKING:
         from sigil_ml.storage import ModelStore

     logger = logging.getLogger(__name__)


     @dataclass
     class _CacheEntry:
         data: bytes
         loaded_at: float


     class ModelCache:
         """In-memory cache wrapping a ModelStore with TTL-based expiration.

         Satisfies the ModelStore protocol so it can be used as a
         drop-in replacement anywhere a ModelStore is expected.
         """

         def __init__(self, inner: "ModelStore", ttl_sec: int = 300) -> None:
             self._inner = inner
             self._ttl = ttl_sec
             self._cache: dict[str, _CacheEntry] = {}
             self._lock = threading.Lock()

         def load(self, model_name: str) -> bytes | None:
             now = time.time()

             with self._lock:
                 entry = self._cache.get(model_name)
                 if entry is not None and (now - entry.loaded_at) < self._ttl:
                     logger.debug("cache hit: %s", model_name)
                     return entry.data

             # Cache miss or expired -- fetch from inner store
             logger.debug("cache miss: %s", model_name)
             data = self._inner.load(model_name)

             if data is not None:
                 with self._lock:
                     self._cache[model_name] = _CacheEntry(
                         data=data, loaded_at=time.time()
                     )

             return data

         def save(self, model_name: str, data: bytes) -> None:
             # Write through to inner store
             self._inner.save(model_name, data)

             # Update cache with fresh data
             with self._lock:
                 self._cache[model_name] = _CacheEntry(
                     data=data, loaded_at=time.time()
                 )

         def evict(self, model_name: str) -> None:
             """Remove a specific entry from the cache.

             Call this when deserialization of cached data fails.
             """
             with self._lock:
                 self._cache.pop(model_name, None)
                 logger.info("cache evicted: %s", model_name)

         def clear(self) -> None:
             """Remove all entries from the cache."""
             with self._lock:
                 self._cache.clear()
                 logger.info("cache cleared")
     ```
  3. Export from `src/sigil_ml/storage/__init__.py`.

- **Files**:
  - Create: `src/sigil_ml/storage/cache.py`
  - Update: `src/sigil_ml/storage/__init__.py` (add export)

- **Parallel?**: No -- core implementation.

- **Notes**:
  - The cache stores bytes, not deserialized model objects. This is intentional: the cache layer is model-agnostic and doesn't need to know about `joblib` or `json`.
  - `load()` releases the lock before calling the inner store's `load()`. This prevents blocking all model loads while one is fetching from S3. The trade-off is that two threads might both fetch the same model on a cache miss -- this is acceptable (idempotent operation, last write to cache wins).
  - `evict()` is a public method so that model classes can call it when deserialization fails.
  - `clear()` is useful for testing and for forced model reloads.

### Subtask T012 -- Add TTL configuration to `src/sigil_ml/config.py`

- **Purpose**: Allow operators to tune cache behavior via environment variable.

- **Steps**:
  1. Verify `config.model_cache_ttl()` exists from WP01/T004. If not present, add it:
     ```python
     def model_cache_ttl() -> int:
         """Return model cache TTL in seconds (default 300 = 5 minutes)."""
         try:
             return int(os.environ.get("SIGIL_MODEL_CACHE_TTL", "300"))
         except ValueError:
             return 300
     ```
  2. This is a verification/completion subtask -- the config was likely added in WP01.

- **Files**:
  - Verify/update: `src/sigil_ml/config.py`

- **Parallel?**: No -- simple verification.

- **Notes**:
  - 300 seconds (5 minutes) is the spec default from FR-008.
  - The TTL is an integer (seconds), not a float. Sub-second precision is unnecessary.
  - Invalid env var values fall back to 300 rather than raising an error.

### Subtask T013 -- Handle multi-tenant cache keys

- **Purpose**: In cloud mode, multiple tenants' models must coexist in the cache without collisions.

- **Steps**:
  1. The cache key is `model_name` as passed to `load()`/`save()`. In cloud mode with per-tenant `S3ModelStore` instances (each with a unique prefix), the cache wraps a single tenant's store, so `model_name` alone is unique.
  2. **However**, if a single `ModelCache` wraps a shared store, or if per-request tenant switching is needed, the key must include the tenant ID.
  3. Preferred approach: **One `ModelCache` per tenant**. The `model_store_factory()` creates a tenant-specific `S3ModelStore`, and the cache wraps it. Cache keys are just `model_name` (e.g., `"stuck"`, `"duration"`).
  4. Alternative (if global cache is preferred): Accept an optional `tenant_id` in `load()` and `save()`, and composite key = `f"{tenant_id}:{model_name}"`. This requires modifying the `ModelStore` protocol, which is undesirable.
  5. **Decision**: Use approach (3) -- one cache per tenant. Document this in the code.

- **Files**:
  - Update: `src/sigil_ml/storage/cache.py` (add docstring clarifying the per-tenant model)

- **Parallel?**: No -- design decision that affects T011.

- **Notes**:
  - This subtask is primarily a design verification. The per-tenant cache model means the factory creates: `ModelCache(S3ModelStore(bucket, tenant_prefix))` for each tenant.
  - For cloud mode with a small number of tenants (< 100), this is memory-efficient. Each tenant gets ~5 models cached.
  - For high-tenant-count deployments, a shared LRU cache may be needed later. Out of scope for this feature.

### Subtask T014 -- Handle cache eviction on deserialization errors

- **Purpose**: When cached model bytes are corrupted (either at source or in transit), the cache must not serve stale/broken data indefinitely.

- **Steps**:
  1. The `ModelCache.evict(model_name)` method (from T011) handles explicit eviction.
  2. Document the eviction pattern for model classes:
     ```python
     # In a model class's __init__ or load method:
     data = self._store.load("stuck")
     if data is not None:
         try:
             self.model = joblib.load(BytesIO(data))
         except Exception:
             logger.warning("Corrupted model data, evicting cache")
             if hasattr(self._store, 'evict'):
                 self._store.evict("stuck")
             self.model = None
     ```
  3. The `evict()` method is only present on `ModelCache`, not on the base `ModelStore` protocol. The `hasattr` check handles both cases gracefully.
  4. After eviction, the next `load()` call will fetch fresh data from the inner store (S3).

- **Files**:
  - Verify: `src/sigil_ml/storage/cache.py`
  - Document pattern for: `src/sigil_ml/models/*.py` (applied in WP04)

- **Parallel?**: No -- depends on T011.

- **Notes**:
  - The eviction pattern is documented here but implemented in model classes during WP04. This subtask ensures the cache supports the pattern.
  - If the fresh data from S3 is also corrupted, the model falls back to rule-based predictions (consistent with spec FR-011).

### Subtask T015 -- Thread-safety for cache reads and writes

- **Purpose**: Cloud mode serves concurrent HTTP requests. Cache operations must be thread-safe to prevent data corruption.

- **Steps**:
  1. Verify the `threading.Lock` usage in T011's implementation.
  2. **Critical pattern**: The lock is released BEFORE calling `self._inner.load()` to avoid blocking all cache operations during a slow S3 fetch:
     ```python
     def load(self, model_name: str) -> bytes | None:
         now = time.time()
         # Check cache under lock
         with self._lock:
             entry = self._cache.get(model_name)
             if entry and (now - entry.loaded_at) < self._ttl:
                 return entry.data
         # Fetch without lock (allows concurrent fetches for different models)
         data = self._inner.load(model_name)
         if data is not None:
             with self._lock:
                 self._cache[model_name] = _CacheEntry(data=data, loaded_at=time.time())
         return data
     ```
  3. This pattern means two threads might both fetch the same model on a concurrent cache miss. This is acceptable: the fetch is idempotent, and the last write to the cache wins.
  4. For higher concurrency control, consider per-key locks:
     ```python
     self._key_locks: dict[str, threading.Lock] = {}
     self._meta_lock = threading.Lock()  # protects _key_locks dict

     def _get_key_lock(self, key: str) -> threading.Lock:
         with self._meta_lock:
             if key not in self._key_locks:
                 self._key_locks[key] = threading.Lock()
             return self._key_locks[key]
     ```
     Only implement per-key locks if the simpler global lock proves to be a bottleneck. Start with the global lock.

- **Files**:
  - Update: `src/sigil_ml/storage/cache.py`

- **Parallel?**: Yes -- can be developed alongside T011-T013 as a refinement.

- **Notes**:
  - Python's GIL provides some protection for dict operations, but explicit locking is still needed for the check-then-act pattern (check TTL, then fetch).
  - The global lock is held only for brief dict operations (microseconds). S3 fetches happen outside the lock. This should not be a bottleneck for typical workloads.

## Risks & Mitigations

- **Risk**: Memory growth from caching many models. **Mitigation**: 5 models per tenant, each typically < 10MB. With 300s TTL, stale entries are replaced naturally. Add a max-entry count later if needed.
- **Risk**: Thundering herd on cache expiry (many requests hit expired entry simultaneously). **Mitigation**: Only one thread fetches from the inner store (per the locking pattern). Others wait or proceed with stale data. Acceptable for this use case.
- **Risk**: Cache returns stale data after training updates the model in S3. **Mitigation**: `save()` updates the cache immediately. TTL ensures other instances pick up changes within 5 minutes.

## Review Guidance

- Verify `ModelCache` satisfies the `ModelStore` protocol (has `load` and `save` with correct signatures).
- Verify TTL logic: cached entries expire after `ttl_sec` seconds.
- Verify `save()` is write-through (updates both cache and inner store).
- Verify `evict()` and `clear()` methods exist for error recovery.
- Verify thread-safety: lock is used for cache dict access, but NOT held during inner store calls.
- Verify the per-tenant cache model is documented clearly.

## Implementation Command

```bash
spec-kitty implement WP03 --base WP01
```

## Activity Log

- 2026-03-29T16:30:00Z -- system -- lane=planned -- Prompt created.
