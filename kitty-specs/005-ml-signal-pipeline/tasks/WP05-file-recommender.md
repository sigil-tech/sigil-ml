---
work_package_id: "WP05"
title: "File Recommender"
lane: "planned"
dependencies: ["WP02"]
subtasks:
  - "T025"
  - "T026"
  - "T027"
  - "T028"
  - "T029"
phase: "Phase 1 - Signal Models"
assignee: ""
agent: ""
shell_pid: ""
review_status: ""
reviewed_by: ""
requirement_refs:
  - "FR-006"
  - "FR-009"
history:
  - timestamp: "2026-03-30T18:27:35Z"
    lane: "planned"
    agent: "system"
    shell_pid: ""
    action: "Prompt generated via /spec-kitty.tasks"
---

# Work Package Prompt: WP05 -- File Recommender

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
spec-kitty implement WP05 --base WP02
```

Depends on WP02 (BehaviorProfile for profile-based filtering). Can run in parallel with WP03 and WP04.

---

## Objectives & Success Criteria

1. `FileRecommender` builds a co-occurrence matrix from completed task file edit patterns.
2. It calculates conditional probabilities: P(file B | file A being edited).
3. Recommendations are scoped to the current repository (no cross-repo recommendations).
4. Signals include structured evidence with recommended files and their co-occurrence scores.
5. The co-occurrence matrix persists via ModelStore and survives restarts.
6. No recommendations are made during cold start (< 5 completed tasks).

## Context & Constraints

- **Spec**: FR-006 (file recommendation from co-occurrence), FR-009 (filtered through behavior profile).
- **Plan**: Design D5 (co-occurrence matrix within task sessions).
- **Research**: R5 confirms approach. 3,808 distinct file paths observed in sample data.
- **Data Model**: Evidence JSON for file_recommender format in data-model.md.
- **DataStore Methods**: `get_completed_task_ids()` and `get_events_for_task()` already exist (see `store.py`).
- **Task Context**: The poller has access to the active task via `store.get_active_task()`. File events include `payload.path`.
- **Acceptance Criteria**: Files A, B, C consistently co-edited -> editing A recommends B and C. Files from other repos are not recommended.

---

## Subtasks & Detailed Guidance

### Subtask T025 -- Create FileRecommender Class

- **Purpose**: Define the `FileRecommender` class skeleton with the `check()` entry point.
- **Steps**:
  1. Create `src/sigil_ml/signals/file_recommender.py`:
     ```python
     """File recommendation based on co-occurrence patterns within task sessions.

     Builds a co-occurrence matrix from completed tasks. When a user edits
     file A, recommends files B, C that are commonly edited alongside A.
     Scoped to the current repository.
     """

     from __future__ import annotations

     import logging
     import os
     from collections import Counter, defaultdict
     from typing import Any

     from sigil_ml.signals import Signal
     from sigil_ml.signals.profile import BehaviorProfile

     logger = logging.getLogger(__name__)

     # Minimum completed tasks before recommendations are made.
     MIN_TASKS_FOR_RECOMMENDATION = 5
     # Minimum co-occurrence probability to trigger a recommendation.
     MIN_COOCCURRENCE_PROB = 0.3
     # Maximum number of files to recommend per signal.
     MAX_RECOMMENDATIONS = 3


     class FileRecommender:
         """Recommends related files based on co-occurrence in task sessions.

         Maintains a sparse co-occurrence matrix built from completed tasks.
         """

         def __init__(self) -> None:
             self._cooccurrence: dict[str, Counter] = defaultdict(Counter)
             self._file_counts: Counter = Counter()
             self._task_count: int = 0

         def check(
             self,
             buffer: list[dict],
             task_context: dict | None,
             profile: BehaviorProfile,
         ) -> list[Signal]:
             """Check for file recommendations based on current editing context.

             Args:
                 buffer: Recent classified events from the poller.
                 task_context: Dict with active task info (may include repo_root).
                 profile: Current user behavior profile.

             Returns:
                 List of Signal objects with file recommendations.
             """
             if self._task_count < MIN_TASKS_FOR_RECOMMENDATION:
                 return []

             current_files = self._extract_current_files(buffer)
             if not current_files:
                 return []

             repo_root = self._infer_repo_root(current_files)
             recommendations = self._recommend(current_files, repo_root)

             if not recommendations:
                 return []

             return [self._build_signal(current_files, recommendations, repo_root)]
     ```
  2. Export from `src/sigil_ml/signals/__init__.py`:
     ```python
     from sigil_ml.signals.file_recommender import FileRecommender
     ```
- **Files**: `src/sigil_ml/signals/file_recommender.py` (new), `src/sigil_ml/signals/__init__.py` (update)
- **Parallel?**: No -- T026-T029 build on this skeleton.
- **Validation**:
  - [ ] `FileRecommender()` constructs without errors
  - [ ] `check([], None, profile)` returns empty list
  - [ ] Returns empty list when task_count < 5

### Subtask T026 -- Co-Occurrence Matrix Builder

- **Purpose**: Build the co-occurrence matrix from completed task event data.
- **Steps**:
  1. Implement `train_from_tasks()`:
     ```python
     def train_from_tasks(self, store) -> int:
         """Build co-occurrence matrix from completed tasks.

         Queries the DataStore for all completed tasks and their events.
         Extracts file pairs edited within the same task.

         Args:
             store: DataStore instance.

         Returns:
             Number of tasks processed.
         """
         self._cooccurrence = defaultdict(Counter)
         self._file_counts = Counter()
         self._task_count = 0

         task_ids = store.get_completed_task_ids()
         for task_id in task_ids:
             events = store.get_events_for_task(task_id)
             files = self._extract_files_from_events(events)
             if len(files) < 2:
                 continue

             self._task_count += 1
             for f in files:
                 self._file_counts[f] += 1
                 for g in files:
                     if f != g:
                         self._cooccurrence[f][g] += 1

         logger.info(
             "FileRecommender: built matrix from %d tasks, %d files",
             self._task_count, len(self._file_counts),
         )
         return self._task_count

     def _extract_files_from_events(self, events: list[dict]) -> set[str]:
         """Extract unique file paths from a list of events."""
         files: set[str] = set()
         for e in events:
             if e.get("kind") != "file":
                 continue
             payload = e.get("payload") or {}
             if isinstance(payload, dict) and "path" in payload:
                 path = str(payload["path"])
                 if path:
                     files.add(path)
         return files
     ```
  2. The matrix is built from ALL completed tasks. For each task, every pair of files edited together increments their co-occurrence count.
  3. `_file_counts` tracks how many tasks each file appears in (denominator for conditional probability).
- **Files**: `src/sigil_ml/signals/file_recommender.py`
- **Parallel?**: Depends on T025 skeleton.
- **Notes**: This queries the DataStore, so it runs during training (WP07), not on every poll cycle. The poller uses the pre-built matrix.
- **Validation**:
  - [ ] After processing tasks where files A,B always co-occur, `_cooccurrence["A"]["B"] > 0`
  - [ ] `_file_counts["A"]` equals the number of tasks containing file A
  - [ ] Tasks with < 2 files are skipped (nothing to co-occur)
  - [ ] Returns count of tasks processed

### Subtask T027 -- Conditional Probability Calculation

- **Purpose**: Compute P(file B | file A) from co-occurrence counts for recommendation scoring.
- **Steps**:
  1. Implement `_recommend()`:
     ```python
     def _recommend(
         self, current_files: set[str], repo_root: str | None
     ) -> list[tuple[str, float]]:
         """Generate file recommendations based on conditional probabilities.

         Args:
             current_files: Set of files currently being edited.
             repo_root: Repository root path for scoping, or None.

         Returns:
             List of (file_path, score) tuples, sorted by score descending.
         """
         candidates: Counter = Counter()

         for f in current_files:
             co = self._cooccurrence.get(f)
             if co is None:
                 continue
             f_count = self._file_counts.get(f, 1)

             for other, count in co.items():
                 # Skip files already being edited
                 if other in current_files:
                     continue
                 # Scope to current repository
                 if repo_root and not other.startswith(repo_root):
                     continue
                 # Conditional probability: P(other | f)
                 prob = count / f_count
                 if prob >= MIN_COOCCURRENCE_PROB:
                     candidates[other] = max(candidates[other], prob)

         return candidates.most_common(MAX_RECOMMENDATIONS)
     ```
  2. The conditional probability `P(B|A) = cooccurrence(A,B) / file_count(A)` represents how often file B was in the same task as file A.
  3. We take the `max` probability across all current files (not sum) to avoid inflating scores when editing many files.
- **Files**: `src/sigil_ml/signals/file_recommender.py`
- **Parallel?**: Depends on T026.
- **Validation**:
  - [ ] Files with high co-occurrence (P > 0.3) are recommended
  - [ ] Files with low co-occurrence (P < 0.3) are not recommended
  - [ ] Files already being edited are excluded
  - [ ] Files outside repo_root are excluded
  - [ ] Returns at most MAX_RECOMMENDATIONS (3) files

### Subtask T028 -- Recommendation Generation Scoped to Current Repo

- **Purpose**: Implement repository scoping and signal generation for file recommendations.
- **Steps**:
  1. Implement helper methods:
     ```python
     def _extract_current_files(self, buffer: list[dict]) -> set[str]:
         """Extract files currently being edited from the event buffer."""
         files: set[str] = set()
         for e in buffer:
             if e.get("kind") != "file":
                 continue
             payload = e.get("payload") or {}
             if isinstance(payload, dict) and "path" in payload:
                 path = str(payload["path"])
                 if path:
                     files.add(path)
         return files

     def _infer_repo_root(self, files: set[str]) -> str | None:
         """Infer the repository root from file paths.

         Looks for common path prefix and checks for .git directory markers.
         Falls back to longest common prefix of all files.
         """
         if not files:
             return None

         paths = sorted(files)
         # Use os.path.commonpath for the common directory prefix
         try:
             common = os.path.commonpath(paths)
             # Walk up to find a likely repo root (directory with common project markers)
             # For now, use the common path as repo root
             return common
         except ValueError:
             # Paths on different drives (Windows) or no common prefix
             return None

     def _build_signal(
         self,
         current_files: set[str],
         recommendations: list[tuple[str, float]],
         repo_root: str | None,
     ) -> Signal:
         """Build a file recommendation signal."""
         # Use the most recently edited file as the trigger
         trigger_file = sorted(current_files)[-1] if current_files else "unknown"

         return Signal(
             signal_type="file_recommendation",
             confidence=round(recommendations[0][1], 4) if recommendations else 0.0,
             evidence={
                 "source_model": "file_recommender",
                 "current_file": trigger_file,
                 "recommended_files": [
                     {"path": path, "co_occurrence": round(score, 4)}
                     for path, score in recommendations
                 ],
                 "context": {
                     "repo": repo_root,
                     "files_being_edited": len(current_files),
                 },
             },
             suggested_action="investigate",
         )
     ```
  2. Repository scoping ensures recommendations stay within the current project. Cross-repo recommendations would be noise.
  3. The `_infer_repo_root()` implementation is simple (common path prefix). It can be enhanced later if needed.
- **Files**: `src/sigil_ml/signals/file_recommender.py`
- **Parallel?**: Partially independent from T027 (different concern: scoping vs probability).
- **Notes**: Evidence JSON matches data-model.md file_recommender format.
- **Validation**:
  - [ ] Files from `/project-A/` are not recommended when editing `/project-B/` files
  - [ ] `_infer_repo_root()` returns a valid directory path
  - [ ] Signal evidence includes recommended_files with path and co_occurrence score
  - [ ] Confidence equals the top recommendation's co-occurrence probability

### Subtask T029 -- Model Persistence via ModelStore

- **Purpose**: Save and load the co-occurrence matrix so the model survives process restarts.
- **Steps**:
  1. Implement save/load methods:
     ```python
     def save(self, model_store) -> None:
         """Persist co-occurrence matrix via ModelStore."""
         import io
         import joblib
         data = {
             "cooccurrence": dict(self._cooccurrence),
             "file_counts": dict(self._file_counts),
             "task_count": self._task_count,
         }
         buf = io.BytesIO()
         joblib.dump(data, buf)
         model_store.save("file_recommender", buf.getvalue())
         logger.info(
             "FileRecommender: saved matrix with %d files, %d tasks",
             len(self._file_counts), self._task_count,
         )

     def load(self, model_store) -> bool:
         """Load co-occurrence matrix from ModelStore. Returns True if loaded."""
         raw = model_store.load("file_recommender")
         if raw is None:
             return False
         import io
         import joblib
         try:
             data = joblib.load(io.BytesIO(raw))
             loaded_co = data.get("cooccurrence", {})
             self._cooccurrence = defaultdict(Counter)
             for f, counts in loaded_co.items():
                 self._cooccurrence[f] = Counter(counts)
             self._file_counts = Counter(data.get("file_counts", {}))
             self._task_count = data.get("task_count", 0)
             logger.info(
                 "FileRecommender: loaded matrix with %d files, %d tasks",
                 len(self._file_counts), self._task_count,
             )
             return True
         except Exception:
             logger.warning("FileRecommender: failed to load model", exc_info=True)
             return False
     ```
  2. Follow the same pattern as NextActionPredictor (T024) and PatternDetector (T018).
  3. Model name is `"file_recommender"`.
- **Files**: `src/sigil_ml/signals/file_recommender.py`
- **Parallel?**: Depends on T025.
- **Validation**:
  - [ ] `save()` followed by `load()` restores identical co-occurrence matrix
  - [ ] `task_count` is preserved across save/load
  - [ ] `load()` returns False when no model exists
  - [ ] `load()` returns False and logs warning on corrupted data

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Large co-occurrence matrix for repos with many files | Medium | Medium | Sparse representation (defaultdict of Counter); only files seen in tasks are tracked |
| Stale recommendations (old file patterns) | Medium | Low | Matrix rebuilt during training cycles (WP07); recent tasks weighted more in future |
| Repo root inference inaccurate | Medium | Low | Simple common-path heuristic; can be enhanced with .git detection later |
| Cross-repo file paths in same task | Low | Low | Scoping filters by repo_root; files outside scope excluded |

## Review Guidance

- **Co-occurrence correctness**: Verify that `_cooccurrence[A][B]` and `_cooccurrence[B][A]` are both incremented (symmetric relationship).
- **Probability calculation**: Verify `P(B|A) = cooccurrence(A,B) / file_count(A)` is computed correctly.
- **Repo scoping**: Verify that `_recommend()` filters by `repo_root` and excludes files already being edited.
- **Cold-start safety**: Verify no recommendations when `task_count < 5`.
- **Evidence structure**: Verify evidence JSON matches data-model.md file_recommender format.
- **DataStore usage**: `train_from_tasks()` uses `store.get_completed_task_ids()` and `store.get_events_for_task()` -- both exist in the protocol.

---

## Activity Log

- 2026-03-30T18:27:35Z -- system -- lane=planned -- Prompt generated via /spec-kitty.tasks.
