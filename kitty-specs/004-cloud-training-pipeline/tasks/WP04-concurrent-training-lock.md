---
work_package_id: "WP04"
title: "Concurrent Training Lock"
lane: "planned"
dependencies: ["WP02"]
subtasks:
  - "T018"
  - "T019"
  - "T020"
  - "T021"
  - "T022"
phase: "Phase 2 - Story Delivery"
assignee: ""
agent: ""
shell_pid: ""
review_status: ""
reviewed_by: ""
requirement_refs:
  - "FR-013"
history:
  - timestamp: "2026-03-30T01:45:09Z"
    lane: "planned"
    agent: "system"
    shell_pid: ""
    action: "Prompt regenerated via /spec-kitty.tasks"
---

# Work Package Prompt: WP04 -- Concurrent Training Lock

## IMPORTANT: Review Feedback Status

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

## Implementation Command

```bash
spec-kitty implement WP04 --base WP02
```

Depends on WP02 (`CloudTrainer.train_tenant()` must exist to integrate with). Can proceed in parallel with WP03.

---

## Objectives & Success Criteria

1. A `TrainingLock` protocol is defined in `src/sigil_ml/training/locking.py` with `acquire(tenant_id) -> bool` and `release(tenant_id)` methods.
2. A `DataStoreTrainingLock` implementation prevents concurrent training for the same tenant using database-level atomicity.
3. `CloudTrainer.train_tenant()` acquires the lock before training and releases in a `finally` block.
4. If the lock cannot be acquired, `train_tenant()` returns a `TrainingRun` with `status="skipped_locked"`.
5. Stale locks (from crashed jobs) older than a configurable timeout (default: 2 hours) are automatically overridden.
6. CloudTrainer works correctly when `training_lock=None` (backward compatible).

## Context & Constraints

- **Spec**: Edge Cases section and FR-013 -- "Concurrent training for the same tenant MUST be prevented."
- **Why distributed locking**: Multiple K8s pods could run training CronJobs concurrently. In-process locks (threading.Lock) are insufficient -- the lock must be at the data layer.
- **WP02 artifacts**: `CloudTrainer.train_tenant()` is the method to integrate with.
- **Simplicity**: Use the existing DataStore (database) for locking rather than introducing Redis/etcd. A simple table-based lock is sufficient.
- **K8s defense-in-depth**: `concurrencyPolicy: Forbid` in the CronJob spec is the first line of defense. Advisory locks are the second.

---

## Subtasks & Detailed Guidance

### Subtask T018 -- Design TrainingLock Protocol

- **Purpose**: Define a clean, minimal protocol for training locks that can be implemented against different backends.
- **Steps**:
  1. Create `src/sigil_ml/training/locking.py`:
     ```python
     """Training lock protocol and implementations for preventing concurrent training."""

     from __future__ import annotations

     import logging
     import os
     from typing import Protocol, runtime_checkable

     logger = logging.getLogger(__name__)

     # Default stale lock timeout: 2 hours (configurable via env var)
     STALE_LOCK_TIMEOUT_SEC = int(os.environ.get("SIGIL_ML_LOCK_TIMEOUT_SEC", "7200"))


     @runtime_checkable
     class TrainingLock(Protocol):
         """Protocol for distributed training locks.

         Prevents concurrent training for the same tenant across
         multiple processes/pods. Implementations must be safe for
         use from multiple K8s pods accessing the same database.
         """

         def acquire(self, tenant_id: str) -> bool:
             """Attempt to acquire a training lock for the given tenant.

             Returns:
                 True if the lock was acquired (caller proceeds with training).
                 False if another process holds a fresh lock (caller should skip).
             """
             ...

         def release(self, tenant_id: str) -> None:
             """Release the training lock for the given tenant.

             No-op if the lock is not held by this process.
             """
             ...
     ```
  2. The protocol is intentionally minimal -- just acquire/release. Stale lock detection is an implementation detail.
  3. `acquire()` returns a boolean (not raises) so the caller can handle the "locked" case gracefully.
- **Files**: `src/sigil_ml/training/locking.py` (new, ~40 lines)
- **Parallel?**: No -- T019 and T020 depend on this.
- **Validation**:
  - [ ] Protocol is importable and runtime-checkable
  - [ ] Method signatures are clear and documented

### Subtask T019 -- Implement DataStoreTrainingLock

- **Purpose**: Implement the lock using the DataStore (database) as the locking backend. Uses an `ml_training_locks` table with atomic INSERT/UPDATE operations.
- **Steps**:
  1. In `locking.py`, implement:
     ```python
     class DataStoreTrainingLock:
         """Training lock backed by a database table via DataStore.

         Uses an ml_training_locks table:
           - tenant_id TEXT PRIMARY KEY
           - acquired_at BIGINT NOT NULL (epoch milliseconds)
           - pid TEXT NOT NULL (process ID of lock holder)

         Acquire logic (atomic):
           1. Try INSERT -- if succeeds, lock acquired
           2. If conflict (lock exists):
              a. If lock.acquired_at + stale_timeout < now -> override (stale)
              b. If fresh -> return False (lock held)

         The DataStore must support:
           - acquire_training_lock(tenant_id, pid, stale_timeout_sec) -> bool
           - release_training_lock(tenant_id) -> None
         """

         def __init__(
             self,
             data_store: "DataStore",
             stale_timeout_sec: int = STALE_LOCK_TIMEOUT_SEC,
         ) -> None:
             self.data_store = data_store
             self.stale_timeout_sec = stale_timeout_sec
             self._pid = str(os.getpid())

         def acquire(self, tenant_id: str) -> bool:
             """Attempt to acquire the lock.

             Handles stale lock detection: if the existing lock is older
             than stale_timeout_sec, it is overridden with a warning.
             """
             acquired = self.data_store.acquire_training_lock(
                 tenant_id=tenant_id,
                 pid=self._pid,
                 stale_timeout_sec=self.stale_timeout_sec,
             )
             if acquired:
                 logger.debug("Acquired training lock for tenant %s (pid=%s)", tenant_id, self._pid)
             else:
                 logger.info("Training lock held for tenant %s, skipping", tenant_id)
             return acquired

         def release(self, tenant_id: str) -> None:
             """Release the lock. No-op if not held."""
             try:
                 self.data_store.release_training_lock(tenant_id)
                 logger.debug("Released training lock for tenant %s", tenant_id)
             except Exception:
                 logger.warning("Failed to release lock for tenant %s", tenant_id, exc_info=True)
     ```
  2. The SQL logic the DataStore must implement:
     ```sql
     -- Acquire: try insert, check stale on conflict
     INSERT INTO ml_training_locks (tenant_id, acquired_at, pid)
     VALUES ($1, $2, $3)
     ON CONFLICT (tenant_id) DO UPDATE
       SET acquired_at = $2, pid = $3
       WHERE ml_training_locks.acquired_at < ($2 - $4 * 1000)
     -- $4 = stale_timeout_sec; returns affected rows > 0 if acquired

     -- Release: simple delete
     DELETE FROM ml_training_locks WHERE tenant_id = $1
     ```
  3. Extend the DataStore Protocol stub with lock methods:
     ```python
     def acquire_training_lock(self, tenant_id: str, pid: str, stale_timeout_sec: int) -> bool: ...
     def release_training_lock(self, tenant_id: str) -> None: ...
     ```
  4. The `ml_training_locks` table schema:
     ```sql
     CREATE TABLE IF NOT EXISTS ml_training_locks (
         tenant_id TEXT PRIMARY KEY,
         acquired_at BIGINT NOT NULL,  -- epoch milliseconds
         pid TEXT NOT NULL              -- process ID
     );
     ```
- **Files**: `src/sigil_ml/training/locking.py` (extend, ~50 lines)
- **Parallel?**: No -- depends on T018.
- **Validation**:
  - [ ] First `acquire("tenant-A")` returns True
  - [ ] Second `acquire("tenant-A")` from same process returns False (lock held)
  - [ ] `release("tenant-A")` followed by `acquire("tenant-A")` returns True
  - [ ] Different tenant IDs do not interfere with each other

### Subtask T020 -- Integrate Lock into train_tenant()

- **Purpose**: Wrap `CloudTrainer.train_tenant()` with lock acquisition and release to prevent concurrent training for the same tenant.
- **Steps**:
  1. Add optional `training_lock` parameter to `CloudTrainer.__init__()`:
     ```python
     class CloudTrainer:
         def __init__(
             self,
             data_store: DataStore,
             model_store: ModelStore,
             config: CloudTrainingConfig | None = None,
             training_lock: TrainingLock | None = None,
         ) -> None:
             self.data_store = data_store
             self.model_store = model_store
             self.config = config or CloudTrainingConfig()
             self.training_lock = training_lock
     ```
  2. In `train_tenant()`, acquire/release the lock as the FIRST operation:
     ```python
     def train_tenant(self, tenant_id: str) -> TrainingRun:
         start = time.time()

         # Lock check (if configured) -- before ANY other operation
         if self.training_lock is not None:
             if not self.training_lock.acquire(tenant_id):
                 return TrainingRun(
                     tenant_id=tenant_id,
                     status="skipped_locked",
                     duration_ms=int((time.time() - start) * 1000),
                 )

         try:
             # ... existing interval check, threshold check, training logic ...
             return run
         finally:
             if self.training_lock is not None:
                 self.training_lock.release(tenant_id)
     ```
  3. Lock check order: Lock -> Interval -> Threshold -> Train. Lock is first because it's the cheapest and prevents wasted work.
  4. The `finally` block ensures the lock is always released, even if training raises an exception.
  5. When `training_lock is None`, the behavior is identical to before -- no locking overhead.
- **Files**: `src/sigil_ml/training/cloud_trainer.py` (modify, ~15 lines)
- **Parallel?**: No -- modifies the core training path.
- **Validation**:
  - [ ] Training without lock configured (`lock=None`) works exactly as before
  - [ ] Training with lock: lock acquired before any data queries
  - [ ] Training failure: lock is released in `finally` block
  - [ ] Lock not acquired: returns `TrainingRun(status="skipped_locked")`
  - [ ] Lock is acquired BEFORE interval check (cheapest check first)

### Subtask T021 -- Add skipped_locked Status to TrainingRun

- **Purpose**: Ensure the `TrainingRun` dataclass and batch summary correctly handle the `"skipped_locked"` status.
- **Steps**:
  1. `TrainingRun.status` is a plain string, so `"skipped_locked"` works without modification. Document the full status vocabulary:
     - `"trained"` -- successfully trained with real data
     - `"skipped"` -- skipped (interval enforcement or insufficient data with no synthetic fallback)
     - `"skipped_locked"` -- skipped because another process holds the training lock
     - `"failed"` -- training failed with an error
  2. Update `TrainingBatch.skipped` property to include `"skipped_locked"`:
     ```python
     @property
     def skipped(self) -> int:
         return sum(1 for r in self.runs if r.status.startswith("skipped"))
     ```
     This already handles `"skipped_locked"` if it uses `startswith("skipped")`.
  3. Verify `TrainingBatch.status_breakdown` (from T017) correctly counts `"skipped_locked"` separately from `"skipped"`.
  4. Add status documentation as a module-level comment or enum-like constants:
     ```python
     # Valid TrainingRun statuses
     STATUS_TRAINED = "trained"
     STATUS_SKIPPED = "skipped"
     STATUS_SKIPPED_LOCKED = "skipped_locked"
     STATUS_FAILED = "failed"
     ```
- **Files**: `src/sigil_ml/training/models.py` (modify, ~10 lines)
- **Parallel?**: Yes -- dataclass changes only.
- **Validation**:
  - [ ] `TrainingRun(tenant_id="x", status="skipped_locked")` serializes correctly
  - [ ] `TrainingBatch.skipped` includes locked-skip count
  - [ ] `status_breakdown` shows `"skipped_locked": N` separately
  - [ ] All status constants are documented

### Subtask T022 -- Stale Lock Detection and Override

- **Purpose**: Prevent permanent deadlocks from crashed training jobs by detecting and overriding stale locks. A lock is stale if it's older than the configurable timeout.
- **Steps**:
  1. Stale detection is built into the `acquire()` SQL logic (see T019):
     ```sql
     ON CONFLICT (tenant_id) DO UPDATE
       SET acquired_at = $now, pid = $pid
       WHERE ml_training_locks.acquired_at < ($now - $stale_timeout_ms)
     ```
     If the WHERE condition matches (lock is stale), the UPDATE succeeds and `acquire()` returns True.
  2. When a stale lock is overridden, the DataStore should return metadata so we can log a warning:
     ```python
     # In DataStoreTrainingLock.acquire():
     if acquired and was_stale_override:
         logger.warning(
             "Overriding stale training lock for tenant %s "
             "(held since %s by pid %s, stale after %ds)",
             tenant_id, old_acquired_at, old_pid, self.stale_timeout_sec,
         )
     ```
  3. The stale timeout is configurable via environment variable:
     ```python
     STALE_LOCK_TIMEOUT_SEC = int(os.environ.get("SIGIL_ML_LOCK_TIMEOUT_SEC", "7200"))
     ```
  4. Stale lock override logging is critical for operational visibility -- it indicates a previous training job crashed.
  5. The default 2-hour timeout is conservative (training typically completes in minutes). Operators can adjust via env var if they have very slow training runs.
- **Files**: `src/sigil_ml/training/locking.py`
- **Parallel?**: No -- extends the lock implementation from T019.
- **Validation**:
  - [ ] Lock acquired 3 hours ago (stale with 2h timeout): overridden on next `acquire()`, returns True
  - [ ] Lock acquired 30 minutes ago (fresh): NOT overridden, `acquire()` returns False
  - [ ] Stale override is logged as a warning (visible in stderr)
  - [ ] Custom timeout via `SIGIL_ML_LOCK_TIMEOUT_SEC=300` overrides default

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Race condition on lock acquire | Low | Medium | Use database-level atomicity (INSERT ON CONFLICT) |
| Lock never released on crash | Low | Medium | Stale lock timeout (default 2h) provides automatic recovery |
| Lock adds latency to training | Very Low | Low | Lock operations are single-row DB queries (~1ms) |
| Over-aggressive stale timeout | Low | Low | Default 2 hours is conservative; configurable via env var |

## Review Guidance

- **Atomicity**: Verify the SQL logic for `acquire()` is truly atomic (single INSERT ON CONFLICT statement, not separate SELECT + INSERT).
- **Finally block**: Verify lock is released in `finally` -- covers both success and exception paths.
- **Backward compatibility**: Verify `training_lock=None` causes zero changes to existing behavior.
- **Lock ordering**: Verify lock is acquired BEFORE interval/threshold checks (prevents wasted work).
- **Stale detection**: Mentally simulate a crashed pod. Verify the stale lock is overridden after timeout.
- **No deadlocks**: Verify a single tenant's lock cannot prevent other tenants from training.

---

## Activity Log

- 2026-03-30T01:45:09Z -- system -- lane=planned -- Prompt regenerated.
