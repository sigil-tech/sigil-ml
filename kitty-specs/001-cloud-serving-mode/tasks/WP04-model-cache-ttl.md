---
work_package_id: WP04
title: Model Cache with TTL and LRU Eviction
lane: planned
dependencies: []
subtasks:
- T018
- T019
- T020
- T021
- T022
- T023
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
- FR-010
---

# Work Package Prompt: WP04 -- Model Cache with TTL and LRU Eviction

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
- Cache entries expire after a configurable TTL (default 300 seconds) using `time.monotonic()`.
- LRU eviction when the cache exceeds `max_size` (default 100 entries).
- Thread-safe for concurrent access from uvicorn's async worker thread pool.
- Cache exposes statistics (hits, misses, evictions) and a `loaded_tenants()` method for observability.
- Configuration via `MODEL_CACHE_TTL_SECONDS` and `MODEL_CACHE_MAX_SIZE` environment variables.
- Factory function `create_model_cache()` for standardized initialization.

**Measurable**:
- `put()` a model, `get()` returns it (cache hit, stats reflect hit).
- Wait past TTL, `get()` returns `None` (cache miss, stats reflect miss+eviction).
- Put 101 models with `max_size=100`, verify oldest is evicted.
- `stats()` returns accurate hit/miss/eviction counts.
- `loaded_tenants()` returns only non-expired entries.

## Context & Constraints

- **Spec**: FR-010 (cache loaded model weights with configurable TTL)
- **Plan**: Design Decision D4 (model cache configuration)
- **Data Model**: `data-model.md` -- ModelCache, ModelCacheEntry, CacheStats entities
- **Research**: R3 (model cache design -- custom dict + lock, not cachetools)

**Constraints**:
- Pure Python only. No external dependencies (no Redis, no cachetools).
- `time.monotonic()` for TTL (immune to system clock changes).
- `threading.Lock` (not `asyncio.Lock`) because model loading may happen in thread pool executors.
- This WP creates a standalone utility module. No integration with endpoints yet (that is WP05).

**Implementation command**: `spec-kitty implement WP04`

## Subtasks & Detailed Guidance

### Subtask T018 -- Create `ModelCache` class with `get()` / `put()` / `evict()` / `evict_all()`

- **Purpose**: Provide the core cache data structure with all public operations.
- **Steps**:
  1. Create `src/sigil_ml/cache.py`:
     ```python
     """In-memory model cache with TTL expiration and LRU eviction.

     Stores loaded model objects keyed by (tenant_id, model_name) with
     time-based expiration. Thread-safe for concurrent async access.
     """

     from __future__ import annotations

     import logging
     import os
     import threading
     import time
     from dataclasses import dataclass
     from typing import Any

     logger = logging.getLogger(__name__)

     # Type alias for cache key.
     CacheKey = tuple[str, str]  # (tenant_id, model_name)


     @dataclass
     class _CacheEntry:
         """Internal: a single cached model with creation timestamp."""

         model: Any
         loaded_at: float  # time.monotonic() timestamp


     class ModelCache:
         """Thread-safe in-memory cache for tenant model objects.

         Args:
             ttl_seconds: Time-to-live for cache entries in seconds.
             max_size: Maximum number of entries before LRU eviction.
         """

         def __init__(
             self,
             ttl_seconds: float = 300.0,
             max_size: int = 100,
         ) -> None:
             self._entries: dict[CacheKey, _CacheEntry] = {}
             self._ttl_seconds = ttl_seconds
             self._max_size = max_size
             self._lock = threading.Lock()
             self._hits = 0
             self._misses = 0
             self._evictions = 0
     ```
  2. Implement `get()`:
     ```python
     def get(self, tenant_id: str, model_name: str) -> Any | None:
         """Retrieve a model from cache.

         Returns the model object on hit, None on miss or expiry.
         """
         key = (tenant_id, model_name)
         with self._lock:
             entry = self._entries.get(key)
             if entry is None:
                 self._misses += 1
                 return None
             age = time.monotonic() - entry.loaded_at
             if age >= self._ttl_seconds:
                 del self._entries[key]
                 self._misses += 1
                 self._evictions += 1
                 return None
             self._hits += 1
             return entry.model
     ```
  3. Implement `put()`:
     ```python
     def put(
         self, tenant_id: str, model_name: str, model: Any
     ) -> None:
         """Store a model in the cache.

         If the cache is at capacity and the key is new, the oldest
         entry is evicted first (LRU).
         """
         key = (tenant_id, model_name)
         entry = _CacheEntry(model=model, loaded_at=time.monotonic())
         with self._lock:
             if key not in self._entries and len(self._entries) >= self._max_size:
                 self._evict_oldest_unlocked()
             self._entries[key] = entry
     ```
  4. Implement `evict()` and `evict_all()`:
     ```python
     def evict(self, tenant_id: str) -> int:
         """Remove all entries for a tenant. Returns count removed."""
         with self._lock:
             keys = [k for k in self._entries if k[0] == tenant_id]
             for k in keys:
                 del self._entries[k]
             self._evictions += len(keys)
             return len(keys)

     def evict_all(self) -> int:
         """Clear the entire cache. Returns count removed."""
         with self._lock:
             count = len(self._entries)
             self._entries.clear()
             self._evictions += count
             return count
     ```
- **Files**: `src/sigil_ml/cache.py` (new file, ~100 lines)
- **Parallel?**: No -- T019-T023 extend this.
- **Notes**: The `_CacheEntry` dataclass is private (underscore prefix) -- only `ModelCache` uses it. Cache values are model objects (e.g., `StuckPredictor` instances, typically 1-50 MB each).

### Subtask T019 -- Implement TTL expiration logic using `time.monotonic()`

- **Purpose**: Ensure expired entries are removed on access and not served to callers.
- **Steps**:
  1. The per-entry TTL check is already in `get()` (T018): compare `time.monotonic() - entry.loaded_at` against `self._ttl_seconds`.
  2. Add a bulk cleanup method for optional periodic maintenance:
     ```python
     def cleanup_expired(self) -> int:
         """Remove all expired entries. Returns count removed.

         This is optional -- expiry is also checked lazily on get().
         Useful for periodic maintenance to free memory.
         """
         now = time.monotonic()
         with self._lock:
             expired = [
                 k for k, v in self._entries.items()
                 if (now - v.loaded_at) >= self._ttl_seconds
             ]
             for k in expired:
                 del self._entries[k]
             self._evictions += len(expired)
             return len(expired)
     ```
  3. Expiration is checked lazily on `get()` (primary mechanism) and can be triggered explicitly via `cleanup_expired()` (secondary mechanism for memory reclamation).
- **Files**: `src/sigil_ml/cache.py` (modify -- add ~15 lines)
- **Parallel?**: No -- builds on T018.
- **Notes**: `time.monotonic()` is immune to wall-clock adjustments (NTP jumps, DST changes). This is important for TTL accuracy in production.

### Subtask T020 -- Implement LRU eviction when cache exceeds `max_size`

- **Purpose**: Prevent unbounded memory growth by evicting the oldest entry when the cache is full.
- **Steps**:
  1. Add the private eviction helper (called within the lock by `put()`):
     ```python
     def _evict_oldest_unlocked(self) -> None:
         """Evict the entry with the oldest loaded_at timestamp.

         MUST be called while holding self._lock.
         """
         if not self._entries:
             return
         oldest_key = min(
             self._entries, key=lambda k: self._entries[k].loaded_at
         )
         del self._entries[oldest_key]
         self._evictions += 1
         logger.debug("cache: evicted oldest entry %s", oldest_key)
     ```
  2. This is already called by `put()` in T018 when `len(self._entries) >= self._max_size`.
  3. With 5 models per tenant and `max_size=100`, this supports ~20 concurrent tenants. Operators can increase via env var.
- **Files**: `src/sigil_ml/cache.py` (modify -- add ~10 lines)
- **Parallel?**: No -- builds on T018.

### Subtask T021 -- Configurable TTL and max_size via environment variables

- **Purpose**: Allow operators to tune cache behavior without code changes (K8s ConfigMaps).
- **Steps**:
  1. Add constants and a factory function at the module level:
     ```python
     DEFAULT_TTL_SECONDS = 300.0
     DEFAULT_MAX_SIZE = 100


     def create_model_cache() -> ModelCache:
         """Create a ModelCache with settings from environment variables.

         Env vars:
             MODEL_CACHE_TTL_SECONDS: TTL in seconds (default 300).
             MODEL_CACHE_MAX_SIZE: Maximum entries (default 100).
         """
         ttl = float(
             os.environ.get("MODEL_CACHE_TTL_SECONDS", str(DEFAULT_TTL_SECONDS))
         )
         max_size = int(
             os.environ.get("MODEL_CACHE_MAX_SIZE", str(DEFAULT_MAX_SIZE))
         )
         logger.info("cache: created with ttl=%.0fs, max_size=%d", ttl, max_size)
         return ModelCache(ttl_seconds=ttl, max_size=max_size)
     ```
  2. WP05 will call `create_model_cache()` when initializing `AppState` in cloud mode.
- **Files**: `src/sigil_ml/cache.py` (modify -- add ~15 lines)
- **Parallel?**: Yes -- independent utility function.
- **Notes**: The env var names `MODEL_CACHE_TTL_SECONDS` and `MODEL_CACHE_MAX_SIZE` match the plan (Design Decision D4).

### Subtask T022 -- Cache statistics tracking: hits, misses, evictions

- **Purpose**: Expose metrics for the `/status` endpoint (WP06) and operational debugging.
- **Steps**:
  1. Implement the `stats()` method:
     ```python
     def stats(self) -> dict[str, Any]:
         """Return cache statistics for observability."""
         with self._lock:
             total = self._hits + self._misses
             return {
                 "entries": len(self._entries),
                 "max_size": self._max_size,
                 "ttl_seconds": self._ttl_seconds,
                 "hits": self._hits,
                 "misses": self._misses,
                 "evictions": self._evictions,
                 "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
             }
     ```
  2. Add a method to list loaded tenants (used by WP06's `/status` and `/health`):
     ```python
     def loaded_tenants(self) -> dict[str, list[str]]:
         """Return mapping of tenant_id -> list of cached model names.

         Only includes non-expired entries.
         """
         now = time.monotonic()
         with self._lock:
             result: dict[str, list[str]] = {}
             for (tenant_id, model_name), entry in self._entries.items():
                 if (now - entry.loaded_at) < self._ttl_seconds:
                     result.setdefault(tenant_id, []).append(model_name)
             return result
     ```
- **Files**: `src/sigil_ml/cache.py` (modify -- add ~25 lines)
- **Parallel?**: Yes -- independent of T023.
- **Notes**: Stats are in-memory and reset on restart. Acceptable for a stateless cloud service.

### Subtask T023 -- Thread-safety with `threading.Lock`

- **Purpose**: Uvicorn runs async handlers via a thread pool. The cache must be safe for concurrent reads and writes.
- **Steps**:
  1. The `threading.Lock` is already initialized in T018's constructor as `self._lock`.
  2. **Verification**: Ensure ALL methods that access `self._entries` or statistics counters acquire the lock:
     - `get()` -- acquires lock (T018)
     - `put()` -- acquires lock (T018)
     - `evict()` -- acquires lock (T018)
     - `evict_all()` -- acquires lock (T018)
     - `cleanup_expired()` -- acquires lock (T019)
     - `stats()` -- acquires lock (T022)
     - `loaded_tenants()` -- acquires lock (T022)
     - `_evict_oldest_unlocked()` -- called within lock by `put()` (T020)
  3. Ensure `_evict_oldest_unlocked()` does NOT acquire the lock (it is called while the lock is already held -- acquiring would deadlock).
  4. Add a docstring note on `_evict_oldest_unlocked()`: "Must be called while holding self._lock."
- **Files**: `src/sigil_ml/cache.py` (verify/audit)
- **Parallel?**: Yes -- verification task.
- **Notes**: `threading.Lock` (not `asyncio.Lock`) is correct because model loading happens in `run_in_executor()` thread pool contexts. `threading.Lock` works across both sync and async code paths.

## Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Memory growth | Low | Medium | LRU eviction at `max_size` (default 100). TTL prevents indefinite staleness. Configurable via env var. |
| Thundering herd on cache miss | Medium | Low | Not handled in this WP. WP05 should add per-key loading lock. |
| Stale models served | Low | Low | TTL ensures max staleness is `ttl_seconds`. Operators can reduce TTL for faster refresh. |
| Deadlock from nested lock acquisition | Low | High | `_evict_oldest_unlocked()` never acquires the lock itself. Documented clearly. |

## Review Guidance

- Verify ALL public methods acquire the lock before accessing shared state.
- Verify `time.monotonic()` is used everywhere (NOT `time.time()`).
- Verify expired entries are not returned by `get()`.
- Verify LRU eviction: put `max_size + 1` entries, verify the oldest is evicted.
- Verify `stats()` returns accurate counts after a sequence of operations.
- Verify `loaded_tenants()` excludes expired entries.
- Verify no external dependencies are imported (only stdlib + typing).
- Verify `_evict_oldest_unlocked()` does NOT acquire the lock.

## Activity Log

- 2026-03-30T01:45:14Z -- system -- lane=planned -- Prompt regenerated.
