"""Feature extraction for ML models.

All database access goes through the DataStore protocol — no direct sqlite3 usage.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

from sigil_ml.store import DataStore

# --- Activity classification features ---

_EVENT_KINDS = ["file", "process", "hyprland", "git", "terminal", "ai"]


def extract_activity_features(event: dict) -> dict[str, float]:
    """Extract features from a single event for ActivityClassifier ML training.

    Returns a flat dict of floats suitable for sklearn input.
    """
    kind = event.get("kind", "")
    payload = event.get("payload") or {}
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            payload = {}

    features: dict[str, float] = {}

    # One-hot encode event kind.
    for k in _EVENT_KINDS:
        features[f"kind_{k}"] = 1.0 if kind == k else 0.0

    # Payload key presence flags.
    features["has_cmd"] = 1.0 if "cmd" in payload else 0.0
    features["has_path"] = 1.0 if "path" in payload else 0.0
    features["has_exit_code"] = 1.0 if "exit_code" in payload else 0.0
    features["exit_code_nonzero"] = 1.0 if payload.get("exit_code", 0) != 0 else 0.0
    features["has_branch"] = 1.0 if "branch" in payload else 0.0

    # Command type classification for terminal events.
    cmd = str(payload.get("cmd", "")).strip().lower()
    features["cmd_is_test"] = (
        1.0
        if any(cmd.startswith(p) for p in ("pytest", "go test", "npm test", "cargo test", "jest", "vitest", "mocha"))
        else 0.0
    )
    features["cmd_is_build"] = (
        1.0
        if any(cmd.startswith(p) for p in ("go build", "npm run build", "cargo build", "make", "./gradlew"))
        else 0.0
    )
    features["cmd_is_lint"] = (
        1.0 if any(cmd.startswith(p) for p in ("flake8", "pylint", "mypy", "ruff", "go vet")) else 0.0
    )
    features["cmd_is_git"] = 1.0 if cmd.startswith("git ") else 0.0

    # File extension category for file events.
    path = str(payload.get("path", ""))
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    features["ext_code"] = 1.0 if ext in ("py", "go", "js", "ts", "rs", "java", "c", "cpp", "rb", "swift") else 0.0
    features["ext_config"] = 1.0 if ext in ("json", "toml", "yaml", "yml", "xml", "ini", "env") else 0.0
    features["ext_docs"] = 1.0 if ext in ("md", "txt", "rst", "adoc") else 0.0

    return features


def extract_stuck_features(store: DataStore, task_id: str) -> dict[str, float]:
    """Extract features for the stuck predictor.

    Returns:
        Dict with keys: test_failure_count, time_in_phase_sec, edit_velocity,
        file_switch_rate, session_length_sec, time_since_last_commit_sec
    """
    task = store.get_task_by_id(task_id)
    if task is None:
        return {
            "test_failure_count": 0.0,
            "time_in_phase_sec": 0.0,
            "edit_velocity": 0.0,
            "file_switch_rate": 0.0,
            "session_length_sec": 0.0,
            "time_since_last_commit_sec": 0.0,
        }

    events = store.get_events_for_task(task_id)

    now_ms = int(time.time() * 1000)
    started_at = task.get("started_at", now_ms)
    last_active = task.get("last_active", now_ms)

    session_length_sec = max((last_active - started_at) / 1000.0, 1.0)
    test_failure_count = float(task.get("test_fails", 0) or 0)

    # Time in current phase: approximate from last phase-change event or started_at
    phase_start = started_at
    for ev in events:
        if ev.get("kind") == "phase_change":
            phase_start = ev.get("ts", phase_start)
    time_in_phase_sec = (now_ms - phase_start) / 1000.0

    # Edit velocity: count edit events
    edit_events = [e for e in events if e.get("kind") in ("edit", "file_edit", "save")]
    edit_count = len(edit_events)
    session_minutes = max(session_length_sec / 60.0, 1 / 60.0)
    edit_velocity = edit_count / session_minutes

    # File switch rate: distinct files / total edits
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


def extract_duration_features(store: DataStore, task_id: str) -> dict[str, float]:
    """Extract features for the duration estimator.

    Returns:
        Dict with keys: file_count, total_edits, time_of_day_hour, branch_name_length
    """
    task = store.get_task_by_id(task_id)
    if task is None:
        return {
            "file_count": 0.0,
            "total_edits": 0.0,
            "time_of_day_hour": float(time.localtime().tm_hour),
            "branch_name_length": 0.0,
        }

    # File count
    files_map = task.get("files")
    if isinstance(files_map, str):
        try:
            files_map = json.loads(files_map)
        except (json.JSONDecodeError, TypeError):
            files_map = {}
    file_count = float(len(files_map)) if isinstance(files_map, dict) else 0.0

    # Total edits from events
    events = store.get_events_for_task(task_id)
    total_edits = float(len([e for e in events if e.get("kind") in ("edit", "file_edit", "save")]))

    # Time of day
    started_at = task.get("started_at")
    if started_at:
        hour = time.localtime(started_at / 1000.0).tm_hour
    else:
        hour = time.localtime().tm_hour

    # Branch name length
    branch = task.get("branch") or ""
    branch_name_length = float(len(branch))

    return {
        "file_count": file_count,
        "total_edits": total_edits,
        "time_of_day_hour": float(hour),
        "branch_name_length": branch_name_length,
    }


def extract_features_from_buffer(events: list[dict]) -> dict[str, float]:
    """Extract stuck-predictor features from a raw event buffer.

    Used by the poller when no active task_id exists (between tasks,
    idle phases). Returns the same shape as extract_stuck_features so
    it is compatible with StuckPredictor.

    Args:
        events: List of raw event dicts from the polling buffer.
                Each dict has keys: id, kind, source, payload (parsed), ts.
    """
    if not events:
        return {
            "test_failure_count": 0.0,
            "time_in_phase_sec": 0.0,
            "edit_velocity": 0.0,
            "file_switch_rate": 0.0,
            "session_length_sec": 0.0,
            "time_since_last_commit_sec": 0.0,
        }

    now_ms = int(time.time() * 1000)
    first_ts = events[0].get("ts", now_ms)
    last_ts = events[-1].get("ts", now_ms)
    session_length_sec = max((last_ts - first_ts) / 1000.0, 1.0)

    edit_events = [e for e in events if e.get("kind") in ("file", "edit")]
    edits = len(edit_events)
    session_minutes = max(session_length_sec / 60.0, 1 / 60.0)
    edit_velocity = edits / session_minutes

    files_seen: set[str] = set()
    for e in edit_events:
        p = e.get("payload") or {}
        if isinstance(p, dict) and "path" in p:
            files_seen.add(p["path"])
    file_switch_rate = len(files_seen) / max(edits, 1)

    commit_events = [e for e in events if e.get("kind") == "git"]
    if commit_events:
        last_commit_ts = max(e.get("ts", 0) for e in commit_events)
        time_since_last_commit_sec = (now_ms - last_commit_ts) / 1000.0
    else:
        time_since_last_commit_sec = session_length_sec

    terminal_events = [e for e in events if e.get("kind") == "terminal"]
    test_failures = sum(
        1 for e in terminal_events if isinstance(e.get("payload"), dict) and e["payload"].get("exit_code", 0) != 0
    )

    return {
        "test_failure_count": float(test_failures),
        "time_in_phase_sec": session_length_sec,
        "edit_velocity": edit_velocity,
        "file_switch_rate": file_switch_rate,
        "session_length_sec": session_length_sec,
        "time_since_last_commit_sec": time_since_last_commit_sec,
    }


# --- Workflow state features ---

_ACTIVITY_CATEGORIES = [
    "creating",
    "refining",
    "editing",
    "verifying",
    "navigating",
    "researching",
    "integrating",
    "communicating",
    "idle",
]


def extract_workflow_features(classified_events: list[dict], session_info: dict) -> dict[str, float]:
    """Extract window-level features from classified events for WorkflowStatePredictor.

    Args:
        classified_events: Events with '_category' key from ActivityClassifier.
        session_info: Dict with 'session_elapsed_min', 'task_phase', 'test_failures'.

    Returns:
        Flat dict of floats suitable for sklearn input.
    """
    total = len(classified_events)
    features: dict[str, float] = {}

    # Normalized category counts.
    counts: dict[str, int] = {}
    for e in classified_events:
        cat = e.get("_category", "idle")
        counts[cat] = counts.get(cat, 0) + 1

    for cat in _ACTIVITY_CATEGORIES:
        features[f"cat_{cat}"] = counts.get(cat, 0) / max(total, 1)

    # Category entropy (Shannon).
    entropy = 0.0
    for cat in _ACTIVITY_CATEGORIES:
        p = features[f"cat_{cat}"]
        if p > 0:
            entropy -= p * math.log2(p)
    features["category_entropy"] = entropy

    # Event rate (events per minute).
    if total >= 2:
        first_ts = classified_events[0].get("ts", 0)
        last_ts = classified_events[-1].get("ts", 0)
        span_min = max((last_ts - first_ts) / 60000.0, 1 / 60.0)
        features["event_rate"] = total / span_min
    else:
        features["event_rate"] = 0.0

    # Category transition count.
    transitions = 0
    for i in range(1, total):
        prev_cat = classified_events[i - 1].get("_category", "")
        curr_cat = classified_events[i].get("_category", "")
        if prev_cat != curr_cat:
            transitions += 1
    features["category_transitions"] = transitions / max(total - 1, 1)

    # Time in dominant category (fraction).
    if counts:
        dominant_count = max(counts.values())
        features["dominant_fraction"] = dominant_count / max(total, 1)
    else:
        features["dominant_fraction"] = 1.0

    # Recent bias: category counts weighted 2x for last 25% of events.
    quarter = max(total // 4, 1)
    recent_counts: dict[str, int] = {}
    for e in classified_events[-quarter:]:
        cat = e.get("_category", "idle")
        recent_counts[cat] = recent_counts.get(cat, 0) + 1
    for cat in _ACTIVITY_CATEGORIES:
        features[f"recent_{cat}"] = recent_counts.get(cat, 0) / max(quarter, 1)

    # Session info features.
    features["session_elapsed_min"] = session_info.get("session_elapsed_min", 0.0)
    features["test_failures"] = float(session_info.get("test_failures", 0))

    return features


# ---------------------------------------------------------------------------
# Cloud-mode feature extractors: accept pre-queried data (task dict + events)
# instead of a DataStore + task_id. Produce identical output to the DataStore-
# based extractors above.
# ---------------------------------------------------------------------------


def extract_stuck_features_from_data(task: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, float]:
    """Extract stuck features from pre-queried task and events data.

    Same output as extract_stuck_features() but operates on passed-in
    data instead of querying a DataStore. For use with DataStore in cloud mode.
    """
    now_ms = int(time.time() * 1000)
    started_at = task.get("started_at", now_ms)
    last_active = task.get("last_active", now_ms)
    session_length_sec = max((last_active - started_at) / 1000.0, 1.0)
    test_failure_count = float(task.get("test_fails", 0) or 0)

    # Time in current phase: approximate from last phase-change event or started_at
    phase_start = started_at
    for ev in events:
        if ev.get("kind") == "phase_change":
            phase_start = ev.get("ts", phase_start)
    time_in_phase_sec = (now_ms - phase_start) / 1000.0

    # Edit velocity: count edit events
    edit_events = [e for e in events if e.get("kind") in ("edit", "file_edit", "save")]
    edit_count = len(edit_events)
    session_minutes = max(session_length_sec / 60.0, 1 / 60.0)
    edit_velocity = edit_count / session_minutes

    # File switch rate: distinct files / total edits
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


def extract_duration_features_from_data(task: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, float]:
    """Extract duration features from pre-queried data.

    Same output as extract_duration_features() but operates on passed-in
    data instead of querying a DataStore. For use with DataStore in cloud mode.
    """
    # File count
    files_map = task.get("files")
    if isinstance(files_map, str):
        try:
            files_map = json.loads(files_map)
        except (json.JSONDecodeError, TypeError):
            files_map = {}
    file_count = float(len(files_map)) if isinstance(files_map, dict) else 0.0

    # Total edits from events
    total_edits = float(len([e for e in events if e.get("kind") in ("edit", "file_edit", "save")]))

    # Time of day
    started_at = task.get("started_at")
    if started_at:
        hour = time.localtime(started_at / 1000.0).tm_hour
    else:
        hour = time.localtime().tm_hour

    # Branch name length
    branch = task.get("branch") or ""
    branch_name_length = float(len(branch))

    return {
        "file_count": file_count,
        "total_edits": total_edits,
        "time_of_day_hour": float(hour),
        "branch_name_length": branch_name_length,
    }


# --- Composite action token extraction (for signal pipeline) ---


def extract_action_token(event: dict) -> str:
    """Convert a classified event into a composite action token.

    Format: "{category}:{tool}" when tool is identifiable,
            "{category}" when tool is unknown.

    Examples: "verifying:pytest", "editing:py", "integrating:git"
    """
    category = event.get("_category", "idle")
    tool = infer_tool(event)
    return f"{category}:{tool}" if tool else category


def infer_tool(event: dict) -> str | None:
    """Infer the specific tool from an event's payload.

    Returns:
        Tool identifier string, or None if cannot be determined.
    """
    kind = event.get("kind", "")
    payload = event.get("payload") or {}
    if isinstance(payload, str):
        return None

    if kind == "terminal":
        cmd = str(payload.get("cmd", "")).strip().split()
        if cmd:
            return cmd[0].split("/")[-1]
        return None

    if kind == "process":
        comm = str(payload.get("comm", "")).split("/")[-1].strip("()")
        return comm if comm else None

    if kind == "git":
        return "git"

    if kind == "file":
        path = str(payload.get("path", ""))
        if "." in path:
            return path.rsplit(".", 1)[-1].lower()
        return "unknown"

    if kind == "ai":
        source = event.get("source", "")
        return source if source else "ai"

    return None
