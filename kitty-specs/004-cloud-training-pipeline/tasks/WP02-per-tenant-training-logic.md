---
work_package_id: "WP02"
title: "Per-Tenant Training Logic"
lane: "planned"
dependencies: ["WP01"]
subtasks:
  - "T007"
  - "T008"
  - "T009"
  - "T010"
  - "T011"
  - "T012"
phase: "Phase 1 - Foundation"
assignee: ""
agent: ""
shell_pid: ""
review_status: ""
reviewed_by: ""
requirement_refs:
  - "FR-004"
  - "FR-005"
  - "FR-006"
  - "FR-009"
  - "FR-012"
history:
  - timestamp: "2026-03-30T01:45:09Z"
    lane: "planned"
    agent: "system"
    shell_pid: ""
    action: "Prompt regenerated via /spec-kitty.tasks"
---

# Work Package Prompt: WP02 -- Per-Tenant Training Logic

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
spec-kitty implement WP02 --base WP01
```

Depends on WP01 (CloudTrainer skeleton and TrainingRun dataclass).

---

## Objectives & Success Criteria

1. `CloudTrainer.train_tenant(tenant_id)` fully implements per-tenant training:
   - Checks minimum interval (default 1 hour) -- skips if trained too recently
   - Checks data threshold (minimum 10 completed tasks) -- falls back to synthetic data below threshold
   - Extracts features from DataStore for all 5 model types
   - Trains stuck, duration, activity, workflow, and quality models
   - Saves weights via ModelStore with tenant-scoped prefix
   - Records training audit events
   - Returns a complete `TrainingRun` with accurate status, sample_count, models_trained, and duration
2. Existing model `.train()` methods are reused without modification.
3. Local training path (Trainer, TrainingScheduler, features.py) remains completely unchanged.

## Context & Constraints

- **Spec**: User Stories 1 and 4, FR-004, FR-005, FR-006, FR-009, FR-012
- **WP01 artifacts**: `src/sigil_ml/training/cloud_trainer.py` (CloudTrainer skeleton), `src/sigil_ml/training/models.py` (dataclasses)
- **Current Trainer reference**: `src/sigil_ml/training/trainer.py` -- shows local training flow. Methods `_train_stuck()` and `_train_duration()` contain combined query+train logic. CloudTrainer must separate these concerns.
- **Feature extraction**: `src/sigil_ml/features.py` has `extract_stuck_features(db_path, task_id)` and `extract_duration_features(db_path, task_id)` that open SQLite directly. These need DataStore-compatible equivalents that accept pre-queried data.
- **Synthetic generators**: `src/sigil_ml/training/synthetic.py` has `generate_stuck_data(n)` and `generate_duration_data(n)`. Activity, workflow, and quality models need cold-start strategies.
- **Model classes**: `src/sigil_ml/models/` -- `stuck.py` (StuckPredictor), `duration.py` (DurationEstimator), `activity.py` (ActivityClassifier), `workflow.py` (WorkflowStatePredictor), `quality.py` (QualityEstimator). Each has a `.train()` method.
- **Model saving strategy**: Current models call `config.weights_path()` and `joblib.dump()` internally in `.train()`. For cloud, CloudTrainer must separately serialize and save via ModelStore. Let the internal save go to the default path (or suppress it); the ModelStore save is the authoritative cloud path.

### Training Flow Order

The `train_tenant()` method must execute checks and operations in this order:
1. Interval check (cheapest -- avoids all data queries if too recent)
2. Data threshold check (counts completed tasks)
3. Feature extraction (queries events per task)
4. Model training (all 5 types)
5. Weight saving (via ModelStore)
6. Audit event recording

---

## Subtasks & Detailed Guidance

### Subtask T007 -- Data Threshold Check

- **Purpose**: Implement the minimum data threshold: at least 10 completed tasks are required for ML training with real data. Below this threshold, fall back to synthetic data (FR-005).
- **Steps**:
  1. In `CloudTrainer.train_tenant()`, after the interval check (T009), query completed tasks:
     ```python
     tasks = self.data_store.query_completed_tasks(tenant_id)
     has_sufficient_data = len(tasks) >= self.config.min_tasks
     ```
  2. The threshold is configurable via `CloudTrainingConfig.min_tasks` (default: 10, overridable via `SIGIL_ML_TRAIN_MIN_TASKS` env var or `--min-tasks` CLI arg).
  3. If below threshold, set a flag for the synthetic fallback path (T008).
  4. If above threshold, proceed with real data extraction (T010).
  5. Boundary case: exactly `min_tasks` tasks qualifies as sufficient data.
- **Files**: `src/sigil_ml/training/cloud_trainer.py`
- **Parallel?**: Part of `train_tenant()` flow, but the check logic is independent from T009.
- **Validation**:
  - [ ] Tenant with 15 tasks proceeds to real data training
  - [ ] Tenant with 5 tasks triggers synthetic fallback
  - [ ] Tenant with exactly 10 tasks (boundary) proceeds to real data training
  - [ ] Configurable threshold: `--min-tasks 5` lowers the bar to 5

### Subtask T008 -- Cold-Start Synthetic Data Fallback

- **Purpose**: When a tenant has insufficient real data, train models using synthetic data generators so the tenant still gets valid model weights. This matches local cold-start behavior from `Trainer._train_stuck()`.
- **Steps**:
  1. Reuse existing generators from `src/sigil_ml/training/synthetic.py`:
     ```python
     from sigil_ml.training.synthetic import generate_stuck_data, generate_duration_data
     ```
  2. In the synthetic fallback path of `train_tenant()`:
     ```python
     if not has_sufficient_data:
         logger.info("Tenant %s has %d tasks (< %d), using synthetic data",
                     tenant_id, len(tasks), self.config.min_tasks)
         models_trained = []

         # Stuck predictor -- synthetic
         X_stuck, y_stuck = generate_stuck_data(500)
         stuck = StuckPredictor()
         stuck.train(X_stuck, y_stuck)
         self._save_model_to_store("stuck", stuck, tenant_id)
         models_trained.append("stuck")

         # Duration estimator -- synthetic
         X_dur, y_dur = generate_duration_data(500)
         duration = DurationEstimator()
         duration.train(X_dur, y_dur)
         self._save_model_to_store("duration", duration, tenant_id)
         models_trained.append("duration")

         # Activity, workflow, quality: rule-based defaults, no training needed for cold-start
         # These models work with default weights when not explicitly trained

         return TrainingRun(
             tenant_id=tenant_id,
             status="trained",  # spec says "trained using synthetic data"
             sample_count=500,
             models_trained=models_trained,
             duration_ms=int((time.time() - start) * 1000),
         )
     ```
  3. Create a `_save_model_to_store()` helper method:
     ```python
     def _save_model_to_store(self, model_name: str, model: Any, tenant_id: str) -> None:
         """Serialize a trained model and save via ModelStore."""
         import io
         import joblib
         buf = io.BytesIO()
         joblib.dump(model.model, buf)  # .model is the sklearn estimator
         tenant_store = self.model_store.for_tenant(tenant_id)
         tenant_store.save(model_name, buf.getvalue())
     ```
  4. Cold-start strategy for models without synthetic generators:
     - **ActivityClassifier**: Rule-based by default (no ML training needed). Skip training; the model uses heuristic classification.
     - **WorkflowStatePredictor**: Also rule-based by default. Skip training.
     - **QualityEstimator**: Uses weight-based scoring. Skip training; default weights apply.
- **Files**: `src/sigil_ml/training/cloud_trainer.py`
- **Parallel?**: Independent code path from T010 (real data extraction), but both are part of `train_tenant()`.
- **Notes**: The spec acceptance scenario says "models are trained using synthetic data (matching local cold-start behavior)." Reference `Trainer._train_stuck()` lines 70-75 for the existing cold-start pattern.
- **Validation**:
  - [ ] Tenant with 5 tasks gets stuck and duration models trained with synthetic data
  - [ ] Weights are saved via ModelStore (not local filesystem)
  - [ ] TrainingRun.models_trained lists only actually trained models
  - [ ] TrainingRun.sample_count reflects synthetic sample count (500)

### Subtask T009 -- Minimum Interval Enforcement

- **Purpose**: Prevent excessive retraining by skipping tenants trained within the last configurable interval (FR-006). This is the first check in `train_tenant()` because it's the cheapest (avoids all data queries).
- **Steps**:
  1. At the very start of `train_tenant()`, check the interval:
     ```python
     def train_tenant(self, tenant_id: str) -> TrainingRun:
         start = time.time()

         # Interval check (cheapest, do first)
         last_ts = self.data_store.get_last_training_ts(tenant_id)
         if last_ts is not None:
             elapsed_sec = time.time() - (last_ts / 1000.0)
             if elapsed_sec < self.config.min_interval_sec:
                 logger.info(
                     "Skipping tenant %s: trained %d sec ago (interval: %d sec)",
                     tenant_id, int(elapsed_sec), self.config.min_interval_sec,
                 )
                 return TrainingRun(
                     tenant_id=tenant_id,
                     status="skipped",
                     duration_ms=int((time.time() - start) * 1000),
                 )

         # ... rest of training logic ...
     ```
  2. The interval is configurable via `CloudTrainingConfig.min_interval_sec` (default: 3600, overridable via `SIGIL_ML_TRAIN_MIN_INTERVAL` env var or `--min-interval` CLI arg).
  3. `get_last_training_ts()` returns epoch milliseconds from the `ml_training_runs` table (see T012), or `None` if the tenant has never been trained.
  4. If `last_ts is None` (never trained), proceed with training.
- **Files**: `src/sigil_ml/training/cloud_trainer.py`
- **Parallel?**: Independent check, part of `train_tenant()` flow.
- **Validation**:
  - [ ] Tenant trained 30 minutes ago is skipped with `status="skipped"`
  - [ ] Tenant trained 2 hours ago proceeds to training
  - [ ] Tenant never trained (`last_ts is None`) proceeds to training
  - [ ] Custom interval: `--min-interval 300` allows retraining after 5 minutes

### Subtask T010 -- Feature Extraction via DataStore

- **Purpose**: Create DataStore-compatible feature extraction functions that produce the same feature dictionaries as the existing SQLite-based extractors, but accept pre-queried data (task dict + events list) instead of a `db_path`.
- **Steps**:
  1. Add new functions to `src/sigil_ml/features.py` (alongside existing functions, NOT replacing them):
     ```python
     def extract_stuck_features_from_data(
         task: dict[str, Any], events: list[dict[str, Any]]
     ) -> dict[str, float]:
         """Extract stuck features from pre-queried task and events data.

         Same output as extract_stuck_features() but operates on passed-in
         data instead of querying SQLite. For use with DataStore in cloud mode.
         """
         now_ms = int(time.time() * 1000)
         started_at = task.get("started_at", now_ms)
         last_active = task.get("last_active", now_ms)
         session_length_sec = max((last_active - started_at) / 1000.0, 1.0)
         test_failure_count = float(task.get("test_fails", 0) or 0)

         # Time in current phase
         phase_start = started_at
         for ev in events:
             if ev.get("kind") == "phase_change":
                 phase_start = ev.get("ts", phase_start)
         time_in_phase_sec = (now_ms - phase_start) / 1000.0

         # Edit velocity
         edit_events = [e for e in events if e.get("kind") in ("edit", "file_edit", "save")]
         edit_count = len(edit_events)
         session_minutes = max(session_length_sec / 60.0, 1 / 60.0)
         edit_velocity = edit_count / session_minutes

         # File switch rate
         files_in_edits: set[str] = set()
         for ev in edit_events:
             payload = ev.get("payload")
             if isinstance(payload, dict) and "file" in payload:
                 files_in_edits.add(payload["file"])
         file_switch_rate = len(files_in_edits) / max(edit_count, 1)

         # Time since last commit
         commit_events = [e for e in events if e.get("kind") == "commit"]
         if commit_events:
             last_commit_ts = max(e.get("ts", 0) for e in commit_events)
             time_since_last_commit_sec = (now_ms - last_commit_ts) / 1000.0
         else:
             time_since_last_commit_sec = session_length_sec

         return {
             "test_failure_count": test_failure_count,
             "time_in_phase_sec": time_in_phase_sec,
             "edit_velocity": edit_velocity,
             "file_switch_rate": file_switch_rate,
             "session_length_sec": session_length_sec,
             "time_since_last_commit_sec": time_since_last_commit_sec,
         }
     ```
  2. Similarly for duration features:
     ```python
     def extract_duration_features_from_data(
         task: dict[str, Any], events: list[dict[str, Any]]
     ) -> dict[str, float]:
         """Extract duration features from pre-queried data."""
         files_map = task.get("files")
         if isinstance(files_map, str):
             try:
                 files_map = json.loads(files_map)
             except (json.JSONDecodeError, TypeError):
                 files_map = {}
         file_count = float(len(files_map)) if isinstance(files_map, dict) else 0.0

         total_edits = float(len([
             e for e in events if e.get("kind") in ("edit", "file_edit", "save")
         ]))

         started_at = task.get("started_at")
         if started_at:
             hour = time.localtime(started_at / 1000.0).tm_hour
         else:
             hour = time.localtime().tm_hour

         branch = task.get("branch") or ""
         branch_name_length = float(len(branch))

         return {
             "file_count": file_count,
             "total_edits": total_edits,
             "time_of_day_hour": float(hour),
             "branch_name_length": branch_name_length,
         }
     ```
  3. The original `extract_stuck_features(db_path, task_id)` and `extract_duration_features(db_path, task_id)` remain **completely unchanged**.
  4. In `CloudTrainer.train_tenant()`, use the new functions:
     ```python
     from sigil_ml.features import extract_stuck_features_from_data, extract_duration_features_from_data

     X_stuck, y_stuck = [], []
     X_dur, y_dur = [], []
     for task in tasks:
         events = self.data_store.query_events_for_task(tenant_id, task["id"])
         stuck_feats = extract_stuck_features_from_data(task, events)
         dur_feats = extract_duration_features_from_data(task, events)
         # ... build X/y arrays ...
     ```
- **Files**: `src/sigil_ml/features.py` (add ~80 lines, do NOT modify existing functions)
- **Parallel?**: Can proceed alongside T007/T009 since it adds new functions only.
- **Notes**: The logic inside the new functions is nearly identical to the existing extractors. The key difference is that they accept `(task: dict, events: list[dict])` instead of `(db_path, task_id)`.
- **Edge Cases**:
  - Task with no events: returns default values (zeros)
  - Task with no `started_at`: uses `time.time()` as fallback
  - Events with string payloads: parse JSON same as existing extractors
- **Validation**:
  - [ ] Given the same task and events data, `extract_stuck_features_from_data()` produces identical output to `extract_stuck_features()` querying the same data from SQLite
  - [ ] Given the same data, `extract_duration_features_from_data()` matches `extract_duration_features()`
  - [ ] Handles missing/null fields gracefully (returns defaults)
  - [ ] Handles empty event lists

### Subtask T011 -- Train All 5 Model Types Per Tenant

- **Purpose**: Complete the training pipeline to train stuck, duration, activity, workflow, and quality models using real tenant data and save weights via ModelStore.
- **Steps**:
  1. In the real-data path of `train_tenant()`, after feature extraction:
     ```python
     import numpy as np
     from sigil_ml.models.stuck import StuckPredictor, FEATURE_NAMES as STUCK_FEATURES
     from sigil_ml.models.duration import DurationEstimator, FEATURE_NAMES as DURATION_FEATURES

     models_trained = []
     total_samples = 0

     # --- Stuck predictor ---
     X_stuck_list, y_stuck_list = [], []
     for task in tasks:
         events = self.data_store.query_events_for_task(tenant_id, task["id"])
         feats = extract_stuck_features_from_data(task, events)
         x = [feats.get(f, 0.0) for f in STUCK_FEATURES]
         X_stuck_list.append(x)
         stuck = feats["test_failure_count"] > 3 and feats["time_in_phase_sec"] > 600
         y_stuck_list.append(1.0 if stuck else 0.0)

     X_stuck = np.array(X_stuck_list)
     y_stuck = np.array(y_stuck_list)
     stuck_model = StuckPredictor()
     stuck_model.train(X_stuck, y_stuck)
     self._save_model_to_store("stuck", stuck_model, tenant_id)
     models_trained.append("stuck")
     total_samples += len(X_stuck)

     # --- Duration estimator ---
     # Similar pattern using extract_duration_features_from_data
     # ... (see T010 for extraction, then build X/y and train)

     # --- Activity classifier ---
     # Extract activity features from events, label with heuristics
     # Train ActivityClassifier if sufficient labeled events

     # --- Workflow predictor ---
     # Extract workflow features from classified events
     # Train WorkflowStatePredictor if sufficient samples

     # --- Quality estimator ---
     # Extract quality scores from task outcomes
     # Train QualityEstimator if sufficient data
     ```
  2. For activity, workflow, and quality models, refer to:
     - `src/sigil_ml/models/activity.py` -- `ActivityClassifier.train()` interface
     - `src/sigil_ml/models/workflow.py` -- `WorkflowStatePredictor.train()` interface
     - `src/sigil_ml/models/quality.py` -- `QualityEstimator.train()` interface
     - `src/sigil_ml/features.py` -- `extract_activity_features()`, `extract_workflow_features()` for feature schemas
  3. Use `_save_model_to_store()` (from T008) to save each trained model.
  4. If a model fails to train (e.g., insufficient specific data for that model type), log a warning and skip it -- do NOT fail the entire tenant run.
  5. The stuck and duration labeling heuristics match the existing `Trainer._train_stuck()` and `Trainer._train_duration()` logic:
     - Stuck label: `test_failure_count > 3 AND time_in_phase_sec > 600`
     - Duration label: `(completed_at - started_at) / 60000.0` capped at `max(duration, 1.0)`
- **Files**: `src/sigil_ml/training/cloud_trainer.py`
- **Parallel?**: No -- depends on T010 (feature extraction) and T008 (save helper).
- **Validation**:
  - [ ] All 5 models are attempted when sufficient data exists
  - [ ] Models with insufficient model-specific data are skipped (not failed)
  - [ ] `TrainingRun.models_trained` accurately lists only successfully trained models
  - [ ] Weights are saved via ModelStore with correct tenant_id scoping
  - [ ] Total `sample_count` reflects actual training data used

### Subtask T012 -- Record Training Audit Events

- **Purpose**: Record an audit trail of each training run to the `ml_training_runs` table via DataStore, matching the pattern from `TrainingScheduler._log_retrain()` (FR-009).
- **Steps**:
  1. At the end of `train_tenant()`, after training completes (regardless of success or failure):
     ```python
     # Record training run to ml_training_runs
     self.data_store.record_training_event(tenant_id, {
         "kind": "training",
         "status": run.status,
         "sample_count": run.sample_count,
         "models_trained": run.models_trained,
         "duration_ms": run.duration_ms,
         "ts": int(time.time() * 1000),
     })
     ```
  2. The event must be recorded for ALL outcomes:
     - `status="trained"` -- successful training with real data
     - `status="trained"` with synthetic -- cold-start training
     - `status="skipped"` -- interval enforcement
     - `status="failed"` -- training error
  3. The record must be compatible with the `ml_training_runs` table schema from plan.md D6:
     ```sql
     CREATE TABLE IF NOT EXISTS ml_training_runs (
         id SERIAL PRIMARY KEY,
         tenant_id TEXT NOT NULL,
         status TEXT NOT NULL,
         models_trained TEXT,       -- JSON array
         sample_count INTEGER,
         duration_ms INTEGER,
         error_message TEXT,
         started_at BIGINT NOT NULL,
         completed_at BIGINT
     );
     ```
  4. Wrap the recording in a try/except so audit failures don't crash the training:
     ```python
     try:
         self.data_store.record_training_event(tenant_id, event)
     except Exception:
         logger.warning("Failed to record training event for tenant %s", tenant_id, exc_info=True)
     ```
- **Files**: `src/sigil_ml/training/cloud_trainer.py`
- **Parallel?**: No -- final step in `train_tenant()`.
- **Validation**:
  - [ ] Successful training records an audit event with `status="trained"`
  - [ ] Failed training records an audit event with `status="failed"` and error details
  - [ ] Skipped training records an audit event with `status="skipped"`
  - [ ] Audit recording failure does not crash the training run

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Feature extraction refactor introduces bugs | Medium | Medium | Keep original functions untouched; new `_from_data` versions mirror logic exactly |
| Activity/workflow/quality model training needs domain knowledge | Medium | Low | Reference existing model `.train()` interfaces; use bootstrapping for labels |
| Model internal `joblib.dump()` conflicts with ModelStore save | Low | Low | Let internal save go to default path; ModelStore save is authoritative. Both can coexist. |
| DataStore API doesn't match Protocol stubs | Medium | Medium | Protocol stubs document expectations; update when real interface lands |

## Review Guidance

- **Training flow order**: Verify the order is interval check -> threshold check -> feature extraction -> model training -> weight saving -> audit event.
- **Feature extraction correctness**: Compare `extract_stuck_features_from_data()` logic line-by-line against `extract_stuck_features()`. They must produce identical outputs for identical inputs.
- **Cold-start parity**: Compare the synthetic fallback path against `Trainer._train_stuck()` lines 70-75 and `Trainer._train_duration()` lines 114-119. Same generator calls, same sample counts.
- **No local modifications**: Verify `trainer.py`, `scheduler.py`, `app.py`, and existing functions in `features.py` are unchanged.
- **Error isolation**: Each model's training should fail independently. A stuck model failure should not prevent duration model training.

---

## Activity Log

- 2026-03-30T01:45:09Z -- system -- lane=planned -- Prompt regenerated.
