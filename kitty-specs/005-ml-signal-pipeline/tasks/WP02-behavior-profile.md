---
work_package_id: "WP02"
title: "Behavior Profile"
lane: "planned"
dependencies: ["WP01"]
subtasks:
  - "T007"
  - "T008"
  - "T009"
  - "T010"
  - "T011"
  - "T012"
phase: "Phase 0 - Foundation"
assignee: ""
agent: ""
shell_pid: ""
review_status: ""
reviewed_by: ""
requirement_refs:
  - "FR-001"
  - "FR-002"
  - "FR-009"
history:
  - timestamp: "2026-03-30T18:27:35Z"
    lane: "planned"
    agent: "system"
    shell_pid: ""
    action: "Prompt generated via /spec-kitty.tasks"
---

# Work Package Prompt: WP02 -- Behavior Profile

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

Depends on WP01 (Signal dataclass, ml_signals table, DataStore methods).

---

## Objectives & Success Criteria

1. `BehaviorProfile` incrementally builds a per-user behavioral summary from classified event buffers.
2. The profile tracks: tool/application usage frequency, file type distribution, and workflow rhythm metrics (commit cadence, test cadence, edit velocity, session length).
3. Profile serializes to a JSON dict and is stored via `DataStore.insert_prediction()` with `model='profile'`.
4. Profile can be restored from the store on startup without replaying full event history.
5. Exponential decay ensures recent behavior is weighted more heavily than old behavior.
6. All three signal models (WP03, WP04, WP05) consume the profile for personalization.

## Context & Constraints

- **Spec**: FR-001 (per-user profile), FR-002 (incremental updates), FR-009 (filter through profile).
- **Data Model**: See `kitty-specs/005-ml-signal-pipeline/data-model.md` for BehaviorProfile JSON structure.
- **Research**: R2 in research.md confirms profile stored in `ml_predictions` with `model='profile'`.
- **Event Classification**: Events arrive with `_category` and `_category_confidence` keys from `ActivityClassifier.classify()` (see `poller.py` line 79-81). Categories: editing, verifying, navigating, researching, integrating, communicating, idle.
- **Event Kinds**: `file`, `process`, `git`, `terminal`, `ai`, `hyprland`. Payloads vary by kind (see features.py for payload parsing patterns).
- **Existing Pattern**: `ActivityClassifier` in `models/activity.py` demonstrates the cold-start pattern -- rule-based initially, ML upgrade after sufficient data. The profile does not need ML but follows a similar incremental philosophy.
- **Performance**: Profile update must complete in <100ms per poll cycle.

---

## Subtasks & Detailed Guidance

### Subtask T007 -- Create BehaviorProfile Class

- **Purpose**: Define the main `BehaviorProfile` class with its core data structures and the `update()` entry point.
- **Steps**:
  1. Create `src/sigil_ml/signals/profile.py`:
     ```python
     """Per-user behavior profile built incrementally from observed events.

     The profile tracks tool frequencies, file type distributions, and
     workflow rhythm metrics. It serves as the personalization backbone
     for all signal models -- ensuring signals only reference tools
     and patterns the user actually uses.
     """

     from __future__ import annotations

     import time
     from collections import Counter
     from dataclasses import dataclass, field
     from typing import Any


     @dataclass
     class RollingStat:
         """Exponentially weighted rolling mean and variance.

         Updates in O(1) time per observation. Tracks count for
         minimum-observation thresholds.
         """
         mean: float = 0.0
         variance: float = 0.0
         count: int = 0
         _alpha: float = 0.995  # decay factor (~500 event half-life)

         @property
         def std(self) -> float:
             return self.variance ** 0.5

         def update(self, value: float) -> None:
             self.count += 1
             if self.count == 1:
                 self.mean = value
                 self.variance = 0.0
                 return
             diff = value - self.mean
             self.mean += (1 - self._alpha) * diff
             self.variance = self._alpha * (self.variance + (1 - self._alpha) * diff * diff)

         def to_dict(self) -> dict:
             return {"mean": self.mean, "variance": self.variance, "count": self.count}

         @classmethod
         def from_dict(cls, d: dict, alpha: float = 0.995) -> RollingStat:
             return cls(
                 mean=d.get("mean", 0.0),
                 variance=d.get("variance", 0.0),
                 count=d.get("count", 0),
                 _alpha=alpha,
             )


     class BehaviorProfile:
         """Incrementally updated per-user behavioral profile.

         Args:
             decay: Exponential decay factor for rolling statistics.
                    Default 0.995 gives ~500 event half-life.
         """

         def __init__(self, decay: float = 0.995) -> None:
             self._decay = decay
             self.tool_counts: Counter = Counter()
             self.file_type_counts: Counter = Counter()
             self.metrics: dict[str, RollingStat] = {}
             self.active_sources: set[str] = set()
             self.total_events_processed: int = 0
             self._last_commit_ts: int = 0
             self._last_test_ts: int = 0
             self._session_start_ts: int = 0

         def update(self, events: list[dict]) -> None:
             """Incrementally update profile from a buffer of classified events."""
             if not events:
                 return
             for e in events:
                 self._update_tool_count(e)
                 self._update_file_types(e)
                 self._track_sources(e)
             self._update_rhythm_stats(events)
             self.total_events_processed += len(events)
     ```
  2. The `RollingStat` helper is defined in the same file (not a separate module) to keep the package simple.
  3. Export from `src/sigil_ml/signals/__init__.py`:
     ```python
     from sigil_ml.signals.profile import BehaviorProfile, RollingStat
     ```
- **Files**: `src/sigil_ml/signals/profile.py` (new), `src/sigil_ml/signals/__init__.py` (update exports)
- **Parallel?**: No -- T008-T012 build on this skeleton.
- **Validation**:
  - [ ] `BehaviorProfile()` constructs without errors
  - [ ] `profile.update([])` is a no-op
  - [ ] `RollingStat.update(5.0)` updates mean and count correctly

### Subtask T008 -- Incremental Tool Frequency Tracking

- **Purpose**: Track which tools/applications the user uses and how frequently, based on process events and terminal commands.
- **Steps**:
  1. Implement `_update_tool_count()` in `BehaviorProfile`:
     ```python
     def _update_tool_count(self, event: dict) -> None:
         """Update tool frequency counter from a single event."""
         kind = event.get("kind", "")
         payload = event.get("payload") or {}
         if isinstance(payload, str):
             return  # Unparsed payload, skip

         if kind == "process":
             comm = str(payload.get("comm", "")).split("/")[-1].strip("()")
             if comm:
                 self.tool_counts[comm] += 1

         elif kind == "terminal":
             cmd = str(payload.get("cmd", "")).strip().split()[0].split("/")[-1]
             if cmd:
                 self.tool_counts[cmd] += 1

         elif kind == "git":
             self.tool_counts["git"] += 1

         elif kind == "ai":
             source = event.get("source", "ai")
             self.tool_counts[source] += 1
     ```
  2. Tool extraction logic matches `infer_tool()` from plan.md D6. The patterns are:
     - Process events: extract command name from `payload.comm`
     - Terminal events: extract first word of `payload.cmd`
     - Git events: always "git"
     - AI events: use event source field
  3. This same tool inference logic will be reused by `extract_action_token()` in WP04 (T020).
- **Files**: `src/sigil_ml/signals/profile.py`
- **Parallel?**: Yes, can proceed alongside T009/T010 (different tracking dimensions).
- **Validation**:
  - [ ] Process event with `comm: "pytest"` increments `tool_counts["pytest"]`
  - [ ] Terminal event with `cmd: "git status"` increments `tool_counts["git"]`
  - [ ] AI event increments the source tool count
  - [ ] Events with empty/missing payloads are skipped gracefully

### Subtask T009 -- File Type Distribution Tracking

- **Purpose**: Track the distribution of file types/extensions the user works with.
- **Steps**:
  1. Implement `_update_file_types()` in `BehaviorProfile`:
     ```python
     def _update_file_types(self, event: dict) -> None:
         """Update file type distribution from file events."""
         if event.get("kind") != "file":
             return
         payload = event.get("payload") or {}
         if isinstance(payload, str):
             return
         path = str(payload.get("path", ""))
         if not path:
             return
         if "." in path:
             ext = "." + path.rsplit(".", 1)[-1].lower()
         else:
             ext = "other"
         self.file_type_counts[ext] += 1
     ```
  2. The distribution is computed at serialization time by normalizing counts:
     ```python
     total = sum(self.file_type_counts.values()) or 1
     distribution = {ext: count / total for ext, count in self.file_type_counts.most_common(20)}
     ```
  3. Cap at top 20 extensions to prevent unbounded growth.
- **Files**: `src/sigil_ml/signals/profile.py`
- **Parallel?**: Yes, independent from T008/T010.
- **Validation**:
  - [ ] File event with path `src/main.py` tracks `.py`
  - [ ] File event with path `Makefile` (no extension) tracks `other`
  - [ ] Distribution values sum to approximately 1.0

### Subtask T010 -- Workflow Rhythm Metrics

- **Purpose**: Track workflow rhythms: commit cadence, test cadence, edit velocity, and session length using `RollingStat` for each metric.
- **Steps**:
  1. Implement `_update_rhythm_stats()` in `BehaviorProfile`:
     ```python
     def _update_rhythm_stats(self, events: list[dict]) -> None:
         """Compute and update rolling statistics for workflow rhythm metrics."""
         if not events:
             return

         now_ms = int(time.time() * 1000)

         # Track session start
         if self._session_start_ts == 0:
             self._session_start_ts = events[0].get("ts", now_ms)

         # Edit velocity: file events per minute in this buffer
         window_start = events[0].get("ts", now_ms)
         window_end = events[-1].get("ts", now_ms)
         window_min = max((window_end - window_start) / 60000.0, 1 / 60.0)
         edit_count = sum(1 for e in events if e.get("kind") == "file")
         if edit_count > 0:
             velocity = edit_count / window_min
             self._get_stat("edit_velocity").update(velocity)

         # Commit cadence: time between git events (minutes)
         for e in events:
             if e.get("kind") == "git":
                 ts = e.get("ts", now_ms)
                 if self._last_commit_ts > 0:
                     gap_min = (ts - self._last_commit_ts) / 60000.0
                     if gap_min > 0:
                         self._get_stat("commit_cadence").update(gap_min)
                 self._last_commit_ts = ts

         # Test cadence: time between terminal test commands (minutes)
         for e in events:
             if e.get("kind") == "terminal":
                 payload = e.get("payload") or {}
                 if isinstance(payload, dict):
                     cmd = str(payload.get("cmd", "")).lower()
                     if any(cmd.startswith(p) for p in (
                         "pytest", "go test", "npm test", "cargo test",
                         "jest", "vitest", "mocha", "python -m pytest",
                     )):
                         ts = e.get("ts", now_ms)
                         if self._last_test_ts > 0:
                             gap_min = (ts - self._last_test_ts) / 60000.0
                             if gap_min > 0:
                                 self._get_stat("test_cadence").update(gap_min)
                         self._last_test_ts = ts

         # Context switch rate: category transitions per buffer
         transitions = 0
         for i in range(1, len(events)):
             if events[i].get("_category") != events[i - 1].get("_category"):
                 transitions += 1
         if len(events) > 1:
             switch_rate = transitions / (len(events) - 1)
             self._get_stat("context_switch_rate").update(switch_rate)

     def _get_stat(self, name: str) -> RollingStat:
         """Get or create a RollingStat for the given metric name."""
         if name not in self.metrics:
             self.metrics[name] = RollingStat(_alpha=self._decay)
         return self.metrics[name]
     ```
  2. Metrics tracked:
     - `edit_velocity`: file events per minute (edits per minute)
     - `commit_cadence`: minutes between git events
     - `test_cadence`: minutes between test commands
     - `context_switch_rate`: fraction of consecutive events with different categories
  3. These are the metrics that PatternDetector (WP03) will monitor for z-score deviations.
- **Files**: `src/sigil_ml/signals/profile.py`
- **Parallel?**: Partially independent -- depends on T007 skeleton.
- **Notes**: Test command detection reuses the same prefixes as `ActivityClassifier._classify_rules()` in `models/activity.py`. Consider extracting to a shared constant.
- **Validation**:
  - [ ] After processing 100+ events with git events, `commit_cadence` stat has count > 0
  - [ ] Edit velocity stat updates proportionally to file event density
  - [ ] Context switch rate is between 0.0 and 1.0
  - [ ] Metrics use exponential decay (not simple averages)

### Subtask T011 -- Profile Serialization

- **Purpose**: Serialize the profile to a JSON dict for storage in `ml_predictions` and deserialize it on startup to avoid replaying full history.
- **Steps**:
  1. Implement `to_dict()` and `from_dict()` on `BehaviorProfile`:
     ```python
     def to_dict(self) -> dict[str, Any]:
         """Serialize profile to a JSON-compatible dict."""
         total_files = sum(self.file_type_counts.values()) or 1
         return {
             "tool_frequency": dict(self.tool_counts.most_common(50)),
             "file_type_distribution": {
                 ext: round(count / total_files, 4)
                 for ext, count in self.file_type_counts.most_common(20)
             },
             "workflow_rhythms": {
                 name: stat.to_dict() for name, stat in self.metrics.items()
             },
             "active_sources": sorted(self.active_sources),
             "total_events_processed": self.total_events_processed,
             "profile_version": 1,
             "updated_at": int(time.time() * 1000),
         }

     @classmethod
     def from_dict(cls, data: dict, decay: float = 0.995) -> BehaviorProfile:
         """Restore a profile from a serialized dict."""
         profile = cls(decay=decay)
         profile.tool_counts = Counter(data.get("tool_frequency", {}))
         # Reconstruct file type counts from distribution (approximate)
         total_events = data.get("total_events_processed", 0)
         dist = data.get("file_type_distribution", {})
         for ext, frac in dist.items():
             profile.file_type_counts[ext] = int(frac * total_events)
         # Restore rolling stats
         rhythms = data.get("workflow_rhythms", {})
         for name, stat_dict in rhythms.items():
             profile.metrics[name] = RollingStat.from_dict(stat_dict, alpha=decay)
         profile.active_sources = set(data.get("active_sources", []))
         profile.total_events_processed = data.get("total_events_processed", 0)
         return profile
     ```
  2. Add `_track_sources()` helper:
     ```python
     def _track_sources(self, event: dict) -> None:
         """Track which event sources are active."""
         kind = event.get("kind", "")
         if kind:
             self.active_sources.add(kind)
     ```
  3. Profile is written to store via `store.insert_prediction("profile", profile.to_dict(), 1.0, None)` -- no TTL, overwritten on each update.
- **Files**: `src/sigil_ml/signals/profile.py`
- **Parallel?**: Depends on T007 skeleton.
- **Notes**: `from_dict()` is approximate for file type counts since we store fractions, not raw counts. This is acceptable because the profile quickly corrects as new events arrive.
- **Validation**:
  - [ ] `from_dict(profile.to_dict())` produces a profile with same tool_counts
  - [ ] Rolling stat metrics survive round-trip (mean, variance, count preserved)
  - [ ] `profile_version` is always 1 (for future schema evolution)
  - [ ] `to_dict()` output matches data-model.md BehaviorProfile JSON structure

### Subtask T012 -- Exponential Decay for Stale Weights

- **Purpose**: Ensure that tools, file types, and patterns that the user stops using gradually lose weight.
- **Steps**:
  1. The `RollingStat` already uses exponential decay via `_alpha` parameter in `update()`. The variance and mean naturally decay toward recent values.
  2. For `tool_counts` and `file_type_counts` (Counter objects), implement periodic decay:
     ```python
     def apply_decay(self) -> None:
         """Apply exponential decay to frequency counters.

         Call periodically (e.g., every 1000 events) to prevent
         stale tools/patterns from persisting indefinitely.
         """
         decay = self._decay ** 100  # Apply 100 steps of decay at once
         decayed_tools = Counter()
         for tool, count in self.tool_counts.items():
             new_count = int(count * decay)
             if new_count > 0:
                 decayed_tools[tool] = new_count
         self.tool_counts = decayed_tools

         decayed_files = Counter()
         for ext, count in self.file_type_counts.items():
             new_count = int(count * decay)
             if new_count > 0:
                 decayed_files[ext] = new_count
         self.file_type_counts = decayed_files
     ```
  3. Call `apply_decay()` inside `update()` every 1000 events:
     ```python
     def update(self, events):
         # ... existing update logic ...
         self.total_events_processed += len(events)
         if self.total_events_processed % 1000 < len(events):
             self.apply_decay()
     ```
  4. This ensures that a tool used heavily 2 weeks ago but not since will gradually lose its count, while recently active tools maintain their counts.
- **Files**: `src/sigil_ml/signals/profile.py`
- **Parallel?**: Depends on T007-T009.
- **Notes**: The decay factor `0.995^100 = ~0.606`. After 1000 events without seeing a tool, its count drops to ~60% of original. After 5000 events, it drops to ~8%.
- **Validation**:
  - [ ] A tool with count 100 after 1000 events of not being seen has count < 70
  - [ ] A tool actively being used maintains its count (decay offset by new increments)
  - [ ] Tools with count dropping to 0 are removed from the Counter

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Profile diverges after restart (no full replay) | Medium | Low | from_dict() restores metrics; new events quickly correct any drift |
| Counter memory grows unboundedly | Low | Low | Decay removes stale entries; cap at top 50 tools and top 20 extensions |
| Rhythm metrics inaccurate on sparse data | Medium | Low | Minimum observation threshold (count > 50) before models use metrics |
| Decay rate too aggressive/conservative | Medium | Medium | Configurable via decay parameter; default 0.995 is conservative |

## Review Guidance

- **Profile completeness**: Verify all fields from data-model.md BehaviorProfile JSON are present in `to_dict()`.
- **Round-trip fidelity**: `from_dict(to_dict())` must preserve RollingStat mean, variance, count exactly (no floating point drift).
- **Decay correctness**: Verify that `apply_decay()` removes zero-count entries and does not inflate counts.
- **Performance**: `update()` on a 100-event buffer should complete in <10ms (no database queries, O(n) processing).
- **Classification dependency**: Verify that `_update_rhythm_stats()` correctly reads `_category` key set by ActivityClassifier.

---

## Activity Log

- 2026-03-30T18:27:35Z -- system -- lane=planned -- Prompt generated via /spec-kitty.tasks.
