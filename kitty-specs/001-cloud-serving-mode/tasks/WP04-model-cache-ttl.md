---
work_package_id: WP04
title: Model Cache with TTL
lane: planned
dependencies: []
subtasks:
- T017
- T018
- T019
- T020
- T021
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
- FR-010
---

# Work Package Prompt: WP04 -- Model Cache with TTL

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

- Implement a `ModelCache` class that stores model objects keyed by `(tenant_id, model_name)`.
- Cache entries expire after a configurable TTL (default 300 seconds).
- The cache is thread-safe for concurrent access from async workers.
- Cache exposes statistics (hits, misses, evictions) for observability.
- Maximum cache size prevents unbounded memory growth.

**Success gate**: Put a model in the cache, retrieve it (cache hit), wait past TTL, attempt retrieval (cache miss). Verify stats reflect the operations. Verify thread safety under concurrent access.

## Context & Constraints

- **Spec**: FR-010 (cache loaded model weights with configurable TTL).
- **No external dependencies**: Pure Python only (stdlib + typing). No Redis, no external cache.
- **Memory safety**: Production deployments may serve many tenants. A cap on cache size is essential.
- **Consumed by WP05**: The `ModelLoader` + `ModelCache` integration happens in WP05.
- **Model objects**: Cache values are model instances (e.g., `StuckPredictor`, `ActivityClassifier`). These are moderately sized in-memory objects (scikit-learn estimators, typically 1-50 MB each).

## Subtasks & Detailed Guidance

### Subtask T017 -- Create `ModelCache` class in `src/sigil_ml/cache.py`

- **Purpose**: Provide the core cache data structure with `get()`, `put()`, and `evict()` operations.
- **Steps**:
  1. Create `src/sigil_ml/cache.py`:
     ```python
     """In-memory model cache with TTL expiration.

     Stores loaded model objects keyed by (tenant_id, model_name) with
     time-based expiration. Thread-safe for concurrent async access.
     """

     from __future__ import annotations

     import logging
     import os
     import threading
     import time
     from dataclasses import dataclass, field
     from typing import Any

     logger = logging.getLogger(__name__)

     # Cache key type
     CacheKey = tuple[str, str]  # (tenant_id, model_name)


     @dataclass
     class CacheEntry:
         """A single cached model with metadata."""
         value: Any
         created_at: float  # time.monotonic()
         ttl_sec: float

         @property
         def is_expired(self) -> bool:
             return (time.monotonic() - self.created_at) >= self.ttl_sec


     class ModelCache:
         """Thread-safe in-memory cache for tenant model objects.

         Args:
             default_ttl_sec: Default time-to-live for cache entries in seconds.
             max_size: Maximum number of entries. Oldest entries are evicted when full.
         """

         def __init__(
             self,
             default_ttl_sec: float = 300.0,
             max_size: int = 100,
         ) -> None:
             self._entries: dict[CacheKey, CacheEntry] = {}
             self._default_ttl_sec = default_ttl_sec
             self._max_size = max_size
             self._lock = threading.Lock()
             # Statistics
             self._hits = 0
             self._misses = 0
             self._evictions = 0

         def get(self, tenant_id: str, model_name: str) -> Any | None:
             """Retrieve a model from cache. Returns None on miss or expiry."""
             ...

         def put(self, tenant_id: str, model_name: str, value: Any, ttl_sec: float | None = None) -> None:
             """Store a model in cache with optional custom TTL."""
             ...

         def evict(self, tenant_id: str) -> int:
             """Remove all entries for a tenant. Returns count of evicted entries."""
             ...

         def evict_all(self) -> int:
             """Clear the entire cache. Returns count of evicted entries."""
             ...

         def stats(self) -> dict[str, Any]:
             """Return cache statistics."""
             ...
     ```
  2. Implement `get()`:
     ```python
     def get(self, tenant_id: str, model_name: str) -> Any | None:
         key = (tenant_id, model_name)
         with self._lock:
             entry = self._entries.get(key)
             if entry is None:
                 self._misses += 1
                 return None
             if entry.is_expired:
                 del self._entries[key]
                 self._misses += 1
                 self._evictions += 1
                 return None
             self._hits += 1
             return entry.value
     ```
  3. Implement `put()`:
     ```python
     def put(self, tenant_id: str, model_name: str, value: Any, ttl_sec: float | None = None) -> None:
         key = (tenant_id, model_name)
         ttl = ttl_sec if ttl_sec is not None else self._default_ttl_sec
         entry = CacheEntry(value=value, created_at=time.monotonic(), ttl_sec=ttl)
         with self._lock:
             # Evict oldest if at capacity (and not replacing existing key)
             if key not in self._entries and len(self._entries) >= self._max_size:
                 self._evict_oldest_unlocked()
             self._entries[key] = entry
     ```
- **Files**: `src/sigil_ml/cache.py` (new file)
- **Parallel?**: No -- T018 and T019 extend this.
- **Notes**: Use `time.monotonic()` instead of `time.time()` for TTL calculations -- it is not affected by system clock changes.

### Subtask T018 -- Implement TTL expiration logic

- **Purpose**: Ensure expired entries are removed and not served.
- **Steps**:
  1. The `is_expired` property on `CacheEntry` handles individual entry expiration (done in T017).
  2. Add a bulk cleanup method for periodic maintenance:
     ```python
     def cleanup_expired(self) -> int:
         """Remove all expired entries. Returns count removed."""
         with self._lock:
             expired_keys = [k for k, v in self._entries.items() if v.is_expired]
             for k in expired_keys:
                 del self._entries[k]
             self._evictions += len(expired_keys)
             return len(expired_keys)
     ```
  3. Add the LRU-style eviction for capacity management:
     ```python
     def _evict_oldest_unlocked(self) -> None:
         """Evict the oldest entry. Must be called while holding self._lock."""
         if not self._entries:
             return
         oldest_key = min(self._entries, key=lambda k: self._entries[k].created_at)
         del self._entries[oldest_key]
         self._evictions += 1
         logger.debug("cache: evicted oldest entry %s", oldest_key)
     ```
  4. Expiration is checked lazily on `get()` (already in T017) and can be triggered explicitly via `cleanup_expired()`.
- **Files**: `src/sigil_ml/cache.py`
- **Parallel?**: No -- builds on T017.
- **Notes**: Lazy expiration (check on get) is sufficient for this use case. The `cleanup_expired()` method is available for optional periodic cleanup but not required.

### Subtask T019 -- Configurable TTL via environment variable

- **Purpose**: Allow operators to tune cache TTL without code changes (useful in K8s ConfigMaps).
- **Steps**:
  1. Add a factory function that reads the env var:
     ```python
     DEFAULT_TTL_SEC = 300.0
     DEFAULT_MAX_SIZE = 100


     def create_model_cache() -> ModelCache:
         """Create a ModelCache with settings from environment variables.

         Env vars:
             SIGIL_MODEL_CACHE_TTL_SEC: TTL in seconds (default 300).
             SIGIL_MODEL_CACHE_MAX_SIZE: Maximum entries (default 100).
         """
         ttl = float(os.environ.get("SIGIL_MODEL_CACHE_TTL_SEC", str(DEFAULT_TTL_SEC)))
         max_size = int(os.environ.get("SIGIL_MODEL_CACHE_MAX_SIZE", str(DEFAULT_MAX_SIZE)))
         logger.info("cache: created with ttl=%ss, max_size=%d", ttl, max_size)
         return ModelCache(default_ttl_sec=ttl, max_size=max_size)
     ```
  2. This factory will be called in WP05 when initializing the cache in `AppState`.
- **Files**: `src/sigil_ml/cache.py`
- **Parallel?**: No -- minor addition.
- **Notes**: Env var names follow the `SIGIL_` prefix convention established in `config.py`.

### Subtask T020 -- Cache statistics: hits, misses, evictions

- **Purpose**: Expose metrics for the `/status` endpoint (WP06) and debugging.
- **Steps**:
  1. Implement the `stats()` method:
     ```python
     def stats(self) -> dict[str, Any]:
         """Return cache statistics for observability."""
         with self._lock:
             total_requests = self._hits + self._misses
             hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
             return {
                 "entries": len(self._entries),
                 "max_size": self._max_size,
                 "default_ttl_sec": self._default_ttl_sec,
                 "hits": self._hits,
                 "misses": self._misses,
                 "evictions": self._evictions,
                 "hit_rate": round(hit_rate, 4),
             }
     ```
  2. Add a method to list loaded tenants:
     ```python
     def loaded_tenants(self) -> dict[str, list[str]]:
         """Return a mapping of tenant_id -> list of cached model names."""
         with self._lock:
             result: dict[str, list[str]] = {}
             for (tenant_id, model_name), entry in self._entries.items():
                 if not entry.is_expired:
                     result.setdefault(tenant_id, []).append(model_name)
             return result
     ```
- **Files**: `src/sigil_ml/cache.py`
- **Parallel?**: Yes (independent of T021).
- **Notes**: These stats are in-memory and reset on restart. This is acceptable for a stateless cloud service.

### Subtask T021 -- Thread-safety with `threading.Lock`

- **Purpose**: FastAPI runs handlers concurrently via asyncio + threadpool. The cache must be safe for concurrent access.
- **Steps**:
  1. The `threading.Lock` is already initialized in T017's constructor.
  2. Verify that ALL methods that access `self._entries` or statistics counters acquire the lock:
     - `get()` -- acquires lock (T017)
     - `put()` -- acquires lock (T017)
     - `evict()` -- must acquire lock
     - `evict_all()` -- must acquire lock
     - `cleanup_expired()` -- acquires lock (T018)
     - `stats()` -- acquires lock (T020)
     - `loaded_tenants()` -- acquires lock (T020)
  3. Implement `evict()` and `evict_all()`:
     ```python
     def evict(self, tenant_id: str) -> int:
         """Remove all entries for a tenant. Returns count of evicted entries."""
         with self._lock:
             keys_to_remove = [k for k in self._entries if k[0] == tenant_id]
             for k in keys_to_remove:
                 del self._entries[k]
             self._evictions += len(keys_to_remove)
             return len(keys_to_remove)

     def evict_all(self) -> int:
         """Clear the entire cache. Returns count of evicted entries."""
         with self._lock:
             count = len(self._entries)
             self._entries.clear()
             self._evictions += count
             return count
     ```
  4. Verify no method reads or writes shared state without the lock.
- **Files**: `src/sigil_ml/cache.py`
- **Parallel?**: Yes (independent of T020).
- **Notes**: `threading.Lock` (not `asyncio.Lock`) is correct here because model loading may happen in thread pool executors. `threading.Lock` works across both sync and async contexts.

## Risks & Mitigations

- **Memory growth**: The `max_size` cap (default 100 entries) prevents unbounded growth. With 5 models per tenant, this supports 20 concurrent tenants. Adjust via env var for larger deployments.
- **Thundering herd**: When a popular tenant's cache entry expires, multiple concurrent requests may all attempt to reload. WP05 should add a loading lock per key to prevent this. The cache itself does not handle this.
- **Stale models**: TTL-based expiration means a model could be stale for up to `ttl_sec` after an update. This is acceptable for ML models which change infrequently.

## Review Guidance

- Verify all public methods acquire the lock before accessing shared state.
- Verify `time.monotonic()` is used (not `time.time()`).
- Verify expired entries are not returned by `get()`.
- Verify `max_size` eviction works (put 101 entries with max_size=100, verify oldest is evicted).
- Verify `stats()` returns accurate counts after a sequence of get/put/evict operations.
- Verify no external dependencies are imported.

## Activity Log

- 2026-03-29T16:29:58Z -- system -- lane=planned -- Prompt created.
