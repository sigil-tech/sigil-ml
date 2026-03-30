---
work_package_id: "WP07"
title: "Training Pipeline Integration"
lane: "planned"
dependencies: ["WP03", "WP04", "WP05"]
subtasks:
  - "T036"
  - "T037"
  - "T038"
  - "T039"
  - "T040"
  - "T041"
phase: "Phase 2 - Integration"
assignee: ""
agent: ""
shell_pid: ""
review_status: ""
reviewed_by: ""
requirement_refs:
  - "FR-011"
  - "FR-012"
  - "FR-013"
  - "FR-014"
  - "FR-015"
history:
  - timestamp: "2026-03-30T18:27:35Z"
    lane: "planned"
    agent: "system"
    shell_pid: ""
    action: "Prompt generated via /spec-kitty.tasks"
---

# Work Package Prompt: WP07 -- Training Pipeline Integration

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
spec-kitty implement WP07 --base WP03,WP04,WP05
```

Depends on WP03, WP04, WP05 (signal model classes must exist for training). Can run in parallel with WP06.

---

## Objectives & Success Criteria

1. `Trainer.train_all()` includes signal model training alongside existing stuck/duration training.
2. Pattern Detector trains an IsolationForest from feedback-labeled signal data.
3. Next-Action Predictor rebuilds complete n-gram tables from all completed task event sequences.
4. File Recommender rebuilds co-occurrence matrix from all completed task file sets.
5. CloudTrainer gains per-tenant and aggregate signal model training.
6. Signal model training follows existing cold-start patterns (synthetic fallback when insufficient data).
7. Existing stuck/duration/activity/workflow/quality training remains unchanged.

## Context & Constraints

- **Spec**: FR-011 (local training from feedback), FR-012 (cloud aggregate training), FR-013 (anonymized patterns only), FR-014 (base model distribution).
- **Plan**: Design D9 (training integration).
- **Research**: R7 (training data requirements): Pattern Detector needs 500+ labeled events; Next-Action needs 10+ completed tasks; File Recommender needs 5+ completed tasks.
- **Trainer**: `src/sigil_ml/training/trainer.py` -- `Trainer.train_all()` calls `_train_stuck()` and `_train_duration()`. Each returns sample count. Pattern: check data threshold, fall back to synthetic if insufficient.
- **CloudTrainer**: `src/sigil_ml/training/cloud_trainer.py` -- `_train_models_from_tasks()` trains stuck and duration. Uses `_save_model_to_store()` for persistence.
- **ModelStore**: Signal models use `model_store.save(name, bytes)` / `model_store.load(name)` for persistence. Names: `"pattern_detector"`, `"next_action"`, `"file_recommender"`.
- **DataStore**: `get_signal_feedback(since_ms)` provides training labels. `get_completed_task_ids()` and `get_events_for_task()` provide training data.

---

## Subtasks & Detailed Guidance

### Subtask T036 -- Add `_train_pattern_detector()` to Trainer

- **Purpose**: Train the IsolationForest from feedback-labeled signal data when sufficient labeled events exist.
- **Steps**:
  1. Add to `src/sigil_ml/training/trainer.py`:
     ```python
     def _train_pattern_detector(self) -> int:
         """Train the PatternDetector's IsolationForest from feedback labels.

         Returns:
             Number of samples used, or 0 if insufficient data.
         """
         # Read feedback data: accepted signals = normal, dismissed = anomalous
         feedback = self.store.get_signal_feedback(since_ms=0)

         if len(feedback) < 500:
             logger.info(
                 "Not enough feedback for pattern detector training (%d, need 500)",
                 len(feedback),
             )
             return 0

         # Build feature matrix from signal evidence
         X_list: list[list[float]] = []
         for fb in feedback:
             # Extract behavioral metrics from the signal evidence
             # The evidence dict contains observed values for metrics
             try:
                 evidence = fb.get("evidence") or {}
                 if isinstance(evidence, str):
                     import json
                     evidence = json.loads(evidence)
                 features = self._extract_pattern_features(evidence)
                 if features is not None:
                     X_list.append(features)
             except Exception:
                 continue

         if len(X_list) < 100:
             logger.info("Not enough valid pattern features (%d)", len(X_list))
             return 0

         import numpy as np
         X = np.array(X_list)

         from sigil_ml.signals.pattern_detector import PatternDetector
         detector = PatternDetector()
         detector.train(X, np.zeros(len(X)))  # IsolationForest is unsupervised
         detector.save(self._model_store)

         return len(X)

     def _extract_pattern_features(self, evidence: dict) -> list[float] | None:
         """Extract a feature vector from signal evidence for IsolationForest training."""
         # Extract numeric values from evidence for multi-dimensional anomaly detection
         source = evidence.get("source_model")
         if source != "pattern_detector":
             return None

         observed = evidence.get("observed")
         baseline_mean = evidence.get("baseline_mean")
         baseline_std = evidence.get("baseline_std")
         z_score = evidence.get("z_score")

         if any(v is None for v in [observed, baseline_mean, baseline_std, z_score]):
             return None

         return [float(observed), float(baseline_mean), float(baseline_std), float(z_score)]
     ```
  2. Training threshold: 500+ labeled feedback events (per research.md R7).
  3. IsolationForest is unsupervised -- it learns the distribution of "normal" behavioral patterns from the feature vectors. Outliers (anomalies) are detected as deviations.
  4. Feature extraction from evidence dicts reuses the structured evidence from PatternDetector (WP03).
- **Files**: `src/sigil_ml/training/trainer.py`
- **Parallel?**: Yes -- independent from T037/T038 (different model types).
- **Validation**:
  - [ ] Returns 0 when fewer than 500 feedback events exist
  - [ ] Trains IsolationForest when sufficient data exists
  - [ ] Saves trained model via ModelStore with name "pattern_detector"
  - [ ] Handles malformed evidence dicts gracefully (skips, does not crash)

### Subtask T037 -- Add `_train_next_action()` to Trainer

- **Purpose**: Rebuild the n-gram model from complete event sequences of all completed tasks.
- **Steps**:
  1. Add to `src/sigil_ml/training/trainer.py`:
     ```python
     def _train_next_action(self) -> int:
         """Rebuild n-gram model from completed task event sequences.

         Returns:
             Number of tokens processed, or 0 if insufficient data.
         """
         task_ids = self.store.get_completed_task_ids()

         if len(task_ids) < 10:
             logger.info(
                 "Not enough completed tasks for next-action training (%d, need 10)",
                 len(task_ids),
             )
             return 0

         from sigil_ml.features import extract_action_token
         from sigil_ml.signals.next_action import NextActionPredictor

         predictor = NextActionPredictor()
         predictor.reset()  # Start fresh for full rebuild
         total_tokens = 0

         for task_id in task_ids:
             events = self.store.get_events_for_task(task_id)
             if not events:
                 continue

             # Classify events (needed for composite tokens)
             from sigil_ml.models.activity import ActivityClassifier
             classifier = ActivityClassifier(model_store=self._model_store)
             for e in events:
                 if "_category" not in e:
                     result = classifier.classify(e)
                     e["_category"] = result["category"]

             tokens = [extract_action_token(e) for e in events]
             predictor.train_incremental(tokens)
             total_tokens += len(tokens)

         if total_tokens > 0:
             predictor.save(self._model_store)

         return total_tokens
     ```
  2. Training threshold: 10+ completed tasks (per research.md R7).
  3. Full rebuild: calls `reset()` first, then processes all task sequences.
  4. Events need `_category` key from ActivityClassifier. If not present (raw events from DB), classify them first.
- **Files**: `src/sigil_ml/training/trainer.py`
- **Parallel?**: Yes -- independent from T036/T038.
- **Notes**: The ActivityClassifier import is inside the method to avoid circular dependencies. It's instantiated fresh for each training run.
- **Validation**:
  - [ ] Returns 0 when fewer than 10 completed tasks exist
  - [ ] Rebuilds n-gram table from scratch (reset + train)
  - [ ] Saves model via ModelStore with name "next_action"
  - [ ] Events without _category are classified before token extraction
  - [ ] Total token count accurately reflects all processed events

### Subtask T038 -- Add `_train_file_recommender()` to Trainer

- **Purpose**: Rebuild the co-occurrence matrix from all completed tasks' file edit patterns.
- **Steps**:
  1. Add to `src/sigil_ml/training/trainer.py`:
     ```python
     def _train_file_recommender(self) -> int:
         """Rebuild co-occurrence matrix from completed task file sets.

         Returns:
             Number of tasks processed, or 0 if insufficient data.
         """
         from sigil_ml.signals.file_recommender import FileRecommender

         recommender = FileRecommender()
         task_count = recommender.train_from_tasks(self.store)

         if task_count < 5:
             logger.info(
                 "Not enough tasks with file data for recommender training (%d, need 5)",
                 task_count,
             )
             return 0

         recommender.save(self._model_store)
         return task_count
     ```
  2. Training threshold: 5+ completed tasks with file data (per research.md R7).
  3. Delegates to `FileRecommender.train_from_tasks(store)` which handles all the matrix building internally (from WP05 T026).
  4. This is the simplest training method because the FileRecommender already has its own training logic.
- **Files**: `src/sigil_ml/training/trainer.py`
- **Parallel?**: Yes -- independent from T036/T037.
- **Validation**:
  - [ ] Returns 0 when fewer than 5 tasks have file data
  - [ ] Delegates to FileRecommender.train_from_tasks()
  - [ ] Saves model via ModelStore with name "file_recommender"
  - [ ] Returns accurate task count

### Subtask T039 -- Integrate Signal Training into Trainer.train_all()

- **Purpose**: Add signal model training to the main `train_all()` orchestration method.
- **Steps**:
  1. Modify `Trainer.train_all()` in `src/sigil_ml/training/trainer.py`:
     ```python
     def train_all(self) -> dict:
         """Train all models and return a summary."""
         start = time.time()
         trained = []
         total_samples = 0

         # Existing model training (unchanged)
         stuck_result = self._train_stuck()
         if stuck_result:
             trained.append("stuck")
             total_samples += stuck_result

         duration_result = self._train_duration()
         if duration_result:
             trained.append("duration")
             total_samples += duration_result

         # Signal model training (additive)
         try:
             pattern_result = self._train_pattern_detector()
             if pattern_result:
                 trained.append("pattern_detector")
                 total_samples += pattern_result
         except Exception:
             logger.warning("Signal model training failed: pattern_detector", exc_info=True)

         try:
             next_action_result = self._train_next_action()
             if next_action_result:
                 trained.append("next_action")
                 total_samples += next_action_result
         except Exception:
             logger.warning("Signal model training failed: next_action", exc_info=True)

         try:
             recommender_result = self._train_file_recommender()
             if recommender_result:
                 trained.append("file_recommender")
                 total_samples += recommender_result
         except Exception:
             logger.warning("Signal model training failed: file_recommender", exc_info=True)

         elapsed = time.time() - start
         return {
             "trained": trained,
             "samples": total_samples,
             "duration_sec": round(elapsed, 2),
         }
     ```
  2. Each signal model training method is wrapped in its own try/except:
     - If pattern_detector training fails, next_action and file_recommender still run
     - If all signal training fails, stuck and duration results are still returned
  3. The return value format is unchanged -- `trained` list gains new model names, `samples` includes signal training samples.
- **Files**: `src/sigil_ml/training/trainer.py`
- **Parallel?**: Depends on T036, T037, T038.
- **Notes**: Signal training failure is a warning, not an error. The existing stuck/duration training path must NEVER be affected by signal training failures.
- **Validation**:
  - [ ] `train_all()` still returns stuck and duration when signal training fails
  - [ ] Signal model names appear in `trained` list when training succeeds
  - [ ] Total sample count includes signal training samples
  - [ ] Each signal model failure is isolated (does not affect others)
  - [ ] Existing `_train_stuck()` and `_train_duration()` are called unchanged

### Subtask T040 -- Add Signal Training to CloudTrainer

- **Purpose**: Extend the CloudTrainer to train signal models alongside existing models for per-tenant and aggregate training.
- **Steps**:
  1. Modify `CloudTrainer._train_models_from_tasks()` in `src/sigil_ml/training/cloud_trainer.py`:
     ```python
     def _train_models_from_tasks(
         self, tasks, task_events, tenant_id
     ) -> list[str]:
         """Train all model types from provided tasks and events."""
         models_trained = []

         # --- Existing models (unchanged) ---
         # ... stuck predictor training ...
         # ... duration estimator training ...

         # --- Signal models (additive) ---

         # Next-Action Predictor: rebuild n-grams from task event sequences
         try:
             from sigil_ml.features import extract_action_token
             from sigil_ml.models.activity import ActivityClassifier
             from sigil_ml.signals.next_action import NextActionPredictor

             predictor = NextActionPredictor()
             predictor.reset()
             total_tokens = 0

             classifier = ActivityClassifier(model_store=self.model_store)
             for task in tasks:
                 events = task_events.get(task["id"], [])
                 for e in events:
                     if "_category" not in e:
                         result = classifier.classify(e)
                         e["_category"] = result["category"]
                 tokens = [extract_action_token(e) for e in events]
                 predictor.train_incremental(tokens)
                 total_tokens += len(tokens)

             if total_tokens > 0:
                 import io
                 import joblib
                 data = {
                     "ngrams": dict(predictor._ngrams),
                     "total_tokens": predictor._total_tokens,
                     "n": predictor._n,
                 }
                 buf = io.BytesIO()
                 joblib.dump(data, buf)
                 scoped_name = f"{tenant_id}/next_action"
                 self.model_store.save(scoped_name, buf.getvalue())
                 models_trained.append("next_action")
         except Exception:
             logger.warning(
                 "Failed to train next_action model for tenant %s",
                 tenant_id, exc_info=True,
             )

         # File Recommender: rebuild co-occurrence from task file sets
         try:
             from sigil_ml.signals.file_recommender import FileRecommender

             recommender = FileRecommender()
             for task in tasks:
                 events = task_events.get(task["id"], [])
                 files = recommender._extract_files_from_events(events)
                 if len(files) < 2:
                     continue
                 recommender._task_count += 1
                 for f in files:
                     recommender._file_counts[f] += 1
                     for g in files:
                         if f != g:
                             recommender._cooccurrence[f][g] += 1

             if recommender._task_count >= 5:
                 import io
                 import joblib
                 data = {
                     "cooccurrence": dict(recommender._cooccurrence),
                     "file_counts": dict(recommender._file_counts),
                     "task_count": recommender._task_count,
                 }
                 buf = io.BytesIO()
                 joblib.dump(data, buf)
                 scoped_name = f"{tenant_id}/file_recommender"
                 self.model_store.save(scoped_name, buf.getvalue())
                 models_trained.append("file_recommender")
         except Exception:
             logger.warning(
                 "Failed to train file_recommender model for tenant %s",
                 tenant_id, exc_info=True,
             )

         return models_trained
     ```
  2. Cloud training uses the same model logic but with tenant-scoped model names and pre-queried data.
  3. Pattern Detector is not trained in cloud mode (requires feedback data which is not pooled across tenants for privacy -- FR-013).
  4. Each signal model failure is isolated, matching the existing stuck/duration error handling pattern.
- **Files**: `src/sigil_ml/training/cloud_trainer.py`
- **Parallel?**: Depends on T036-T038 (model classes must exist).
- **Notes**: The CloudTrainer pattern differs from local Trainer -- it receives pre-queried `tasks` and `task_events` instead of querying the DataStore directly.
- **Validation**:
  - [ ] Per-tenant training includes next_action and file_recommender
  - [ ] Model weights saved with tenant-scoped names (e.g., `tenant-a/next_action`)
  - [ ] Aggregate training includes signal models
  - [ ] Pattern detector is NOT trained in cloud mode (privacy constraint)
  - [ ] Existing stuck/duration training is unchanged

### Subtask T041 -- Synthetic Data Generation for Signal Cold-Start

- **Purpose**: Add synthetic data generators for signal models to enable cold-start training when real data is insufficient.
- **Steps**:
  1. Add to `src/sigil_ml/training/synthetic.py`:
     ```python
     def generate_next_action_data(n: int = 500) -> list[list[str]]:
         """Generate synthetic event token sequences for n-gram cold start.

         Returns:
             List of token sequences (each a list of composite action tokens).
         """
         import random
         rng = random.Random(42)

         # Common workflow patterns
         patterns = [
             ["editing:py", "editing:py", "verifying:pytest", "integrating:git"],
             ["editing:go", "editing:go", "verifying:go", "integrating:git"],
             ["editing:js", "editing:js", "verifying:jest", "integrating:git"],
             ["researching:ai", "editing:py", "editing:py", "verifying:pytest"],
             ["navigating", "editing:py", "editing:py", "editing:py", "verifying:pytest"],
         ]

         sequences: list[list[str]] = []
         for _ in range(n):
             base = rng.choice(patterns)
             # Add some noise: occasionally insert extra editing or navigating
             seq = []
             for token in base:
                 seq.append(token)
                 if rng.random() < 0.2:
                     seq.append(rng.choice(["editing:py", "navigating", "idle"]))
             sequences.append(seq)

         return sequences


     def generate_file_cooccurrence_data(
         n_tasks: int = 50, n_files: int = 20
     ) -> list[set[str]]:
         """Generate synthetic file co-occurrence data for cold start.

         Returns:
             List of file sets (each representing files edited in one task).
         """
         import random
         rng = random.Random(42)

         # Create file clusters that tend to co-occur
         files = [f"src/module_{i}.py" for i in range(n_files)]
         clusters = [
             set(files[0:4]),   # cluster 1
             set(files[4:8]),   # cluster 2
             set(files[8:12]),  # cluster 3
         ]

         tasks: list[set[str]] = []
         for _ in range(n_tasks):
             cluster = rng.choice(clusters)
             # Select 2-4 files from the cluster, plus occasional cross-cluster file
             task_files = set(rng.sample(sorted(cluster), min(rng.randint(2, 4), len(cluster))))
             if rng.random() < 0.15:
                 task_files.add(rng.choice(files))
             tasks.append(task_files)

         return tasks
     ```
  2. These generators produce realistic-looking patterns:
     - N-gram sequences follow common edit-test-commit workflows
     - File co-occurrence data clusters files that would naturally be edited together
  3. CloudTrainer uses these in `_train_synthetic()` alongside existing stuck/duration generators:
     ```python
     def _train_synthetic(self, tenant_id: str) -> list[str]:
         models_trained = []
         # ... existing stuck/duration synthetic training ...

         # Next-action synthetic
         try:
             from sigil_ml.training.synthetic import generate_next_action_data
             from sigil_ml.signals.next_action import NextActionPredictor

             sequences = generate_next_action_data(500)
             predictor = NextActionPredictor()
             for seq in sequences:
                 predictor.train_incremental(seq)
             # Save via ModelStore with tenant prefix
             predictor.save(self.model_store)  # TODO: tenant-scoped save
             models_trained.append("next_action")
         except Exception:
             logger.warning("Failed to train synthetic next_action", exc_info=True)

         return models_trained
     ```
- **Files**: `src/sigil_ml/training/synthetic.py` (add ~60 lines), `src/sigil_ml/training/cloud_trainer.py` (update `_train_synthetic`)
- **Parallel?**: Yes -- independent from T036-T038.
- **Notes**: Synthetic data uses `Random(42)` for deterministic reproducibility (matching existing synthetic generators).
- **Validation**:
  - [ ] `generate_next_action_data(100)` returns 100 token sequences
  - [ ] Sequences contain valid composite action tokens
  - [ ] `generate_file_cooccurrence_data(50)` returns 50 file sets
  - [ ] File sets contain 2+ files each (required for co-occurrence)
  - [ ] RNG seed 42 produces deterministic output

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Training time increase | Low | Low | Signal models are lightweight (n-gram rebuild, sparse matrix); expected <10s additional |
| Insufficient feedback for IsolationForest | Expected | Low | Graceful fallback: return 0 samples, pattern detector stays in z-score mode |
| Event classification overhead in training | Medium | Low | ActivityClassifier is lightweight (rule-based by default); classify each event once |
| Signal training breaks existing training | Low | Critical | Each signal method wrapped in try/except; existing methods called first |

## Review Guidance

- **Non-regression**: Verify `_train_stuck()` and `_train_duration()` are called BEFORE signal training methods. Verify they are completely unchanged.
- **Error isolation**: Verify each signal training method has its own try/except in `train_all()`.
- **Data thresholds**: Verify thresholds match research.md R7: pattern_detector=500 feedback, next_action=10 tasks, file_recommender=5 tasks.
- **Model persistence**: Verify each signal model is saved via ModelStore with correct name.
- **CloudTrainer parity**: Verify cloud training includes next_action and file_recommender but NOT pattern_detector (privacy).
- **Synthetic determinism**: Verify synthetic generators use `Random(42)` for reproducibility.
- **Existing imports**: Verify no new top-level imports are added to `trainer.py` (signal imports are inside methods to avoid circular deps).

---

## Activity Log

- 2026-03-30T18:27:35Z -- system -- lane=planned -- Prompt generated via /spec-kitty.tasks.
