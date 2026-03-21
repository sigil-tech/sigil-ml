"""Event poller — continuous push-to-db prediction loop.

Polls events → classifies activity → runs models → writes to ml_predictions.
Runs as an asyncio background task inside the FastAPI process.
"""

import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path

from sigil_ml.features import (
    extract_duration_features,
    extract_features_from_buffer,
    extract_stuck_features,
)

logger = logging.getLogger(__name__)

POLL_INTERVAL_SEC = 0.5
PREDICT_EVERY_N_EVENTS = 3  # minimum events before predicting
PREDICT_MIN_INTERVAL_SEC = 60  # minimum seconds between prediction cycles
QUALITY_WINDOW_SEC = 1800  # 30-minute rolling window for quality features
PREDICTION_TTL_SEC = 90  # 90-second expiry for stuck/activity/workflow
QUALITY_TTL_SEC = 120  # 2-minute expiry for quality


class EventPoller:
    """Polls sigild's events table and writes predictions to ml_predictions."""

    def __init__(self, db_path: Path, models: dict) -> None:
        self.db_path = db_path
        self.stuck = models["stuck"]
        self.activity = models["activity"]
        self.workflow = models["workflow"]
        self.duration = models["duration"]
        self.quality = models["quality"]
        self._buffer: list[dict] = []
        self._since_last_predict = 0
        self._last_predict_time = 0.0
        self._running = False

    async def run(self) -> None:
        """Main loop — call as an asyncio task."""
        self._running = True
        logger.info("poller: started against %s", self.db_path)
        while self._running:
            try:
                await asyncio.get_event_loop().run_in_executor(None, self._poll_once)
            except sqlite3.OperationalError as e:
                # Database may not exist yet or be locked — retry silently.
                logger.debug("poller: sqlite error (will retry): %s", e)
            except Exception:
                logger.exception("poller: unhandled error")
            await asyncio.sleep(POLL_INTERVAL_SEC)

    def stop(self) -> None:
        self._running = False

    def _poll_once(self) -> None:
        conn = self._connect()
        try:
            cursor_id = conn.execute("SELECT last_event_id FROM ml_cursor WHERE id = 1").fetchone()
            since = cursor_id[0] if cursor_id else 0

            rows = conn.execute(
                "SELECT id, kind, source, payload, ts FROM events WHERE id > ? ORDER BY id ASC LIMIT 100",
                (since,),
            ).fetchall()

            if not rows:
                return

            events = []
            for row in rows:
                e = dict(zip(["id", "kind", "source", "payload", "ts"], row))
                if isinstance(e.get("payload"), str):
                    try:
                        e["payload"] = json.loads(e["payload"])
                    except (json.JSONDecodeError, TypeError):
                        pass

                # Classify each event as it enters the buffer.
                classification = self.activity.classify(e)
                e["_category"] = classification["category"]
                e["_category_confidence"] = classification["confidence"]

                events.append(e)

            self._buffer.extend(events)
            self._buffer = self._buffer[-200:]  # keep last 200
            self._since_last_predict += len(events)

            max_id = max(e["id"] for e in events)
            conn.execute(
                "UPDATE ml_cursor SET last_event_id = ?, updated_at = ? WHERE id = 1",
                (max_id, int(time.time() * 1000)),
            )

            elapsed = time.time() - self._last_predict_time
            if self._since_last_predict >= PREDICT_EVERY_N_EVENTS and elapsed >= PREDICT_MIN_INTERVAL_SEC:
                self._predict_and_write(conn)
                self._since_last_predict = 0
                self._last_predict_time = time.time()

            conn.commit()
        finally:
            conn.close()

    # Fallback predictions for untrained models.
    _FALLBACK_STUCK = {"probability": 0.5, "confidence": "weak"}
    _FALLBACK_WORKFLOW = {
        "flow_state": {
            "deep_work": 0.0,
            "shallow_work": 1.0,
            "exploring": 0.0,
            "blocked": 0.0,
            "winding_down": 0.0,
        },
        "dominant_state": "shallow_work",
        "momentum": 0.0,
        "focus_score": 0.5,
        "dominant_activity": "idle",
        "activity_distribution": {},
        "session_elapsed_min": 0.0,
        "method": "rules",
        "confidence": 0.5,
    }
    _FALLBACK_DURATION = {"estimated_minutes": 60.0, "confidence_interval": [30.0, 90.0]}

    def _predict_and_write(self, conn: sqlite3.Connection) -> None:
        start = time.time()
        task_id = self._active_task_id(conn)

        # Stuck prediction — check is_trained before calling predict.
        if self.stuck.is_trained:
            if task_id:
                feats = extract_stuck_features(self.db_path, task_id)
            else:
                feats = extract_features_from_buffer(self._buffer)
            result = self.stuck.predict(feats)
        else:
            result = self._FALLBACK_STUCK
        self._write(conn, "stuck", result, result.get("probability", 0.5), PREDICTION_TTL_SEC)

        # Activity summary — classify and summarize the buffer.
        activity_result = self._activity_summary()
        self._write(conn, "activity", activity_result, activity_result.get("confidence", 0.5), PREDICTION_TTL_SEC)

        # Workflow state prediction — replaces old suggestion policy.
        session_info = self._session_info(conn, task_id)
        result = self.workflow.predict(self._buffer, session_info)
        self._write(conn, "suggest", result, result.get("confidence", 0.5), PREDICTION_TTL_SEC)

        # Duration — only when active task AND model is trained.
        if task_id and self.duration.is_trained:
            try:
                feats = extract_duration_features(self.db_path, task_id)
                result = self.duration.predict(feats)
                ci = result.get("confidence_interval", [30, 90])
                est = result.get("estimated_minutes", 60)
                rel_width = (ci[1] - ci[0]) / max(est, 1.0)
                conf = max(0.0, min(1.0, 1.0 - rel_width / 2.0))
                self._write(conn, "duration", result, conf, None)
            except Exception:
                logger.debug("poller: duration prediction skipped", exc_info=True)
        elif task_id:
            self._write(conn, "duration", self._FALLBACK_DURATION, 0.5, None)

        # Quality score — always callable (rule-based, no training required).
        qfeats = self._quality_features(conn)
        result = self.quality.predict(qfeats)
        self._write(conn, "quality", result, result.get("score", 50) / 100.0, QUALITY_TTL_SEC)

        # Audit log
        latency_ms = int((time.time() - start) * 1000)
        conn.execute(
            "INSERT INTO ml_events (kind, endpoint, routing, latency_ms, ts) "
            "VALUES ('prediction', 'poller', 'local', ?, ?)",
            (latency_ms, int(time.time() * 1000)),
        )

    def _activity_summary(self) -> dict:
        """Build activity summary from classified buffer events."""
        window_summary: dict[str, int] = {}
        for e in self._buffer:
            cat = e.get("_category", "idle")
            window_summary[cat] = window_summary.get(cat, 0) + 1

        recent = [
            {"ts": e.get("ts", 0), "kind": e.get("kind", ""), "category": e.get("_category", "idle")}
            for e in self._buffer[-10:]
        ]

        dominant = max(window_summary, key=window_summary.get) if window_summary else "idle"

        # Average confidence of classifications.
        confidences = [e.get("_category_confidence", 0.5) for e in self._buffer]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.5

        return {
            "window_summary": window_summary,
            "recent": recent,
            "dominant": dominant,
            "method": "rules",
            "confidence": round(avg_conf, 4),
        }

    def _session_info(self, conn: sqlite3.Connection, task_id: str | None) -> dict:
        """Build session info for WorkflowStatePredictor."""
        session_elapsed_min = 0.0
        task_phase = None
        test_failures = 0

        if task_id:
            row = conn.execute(
                "SELECT started_at, phase, test_fails FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
            if row:
                started_at = row[0] or 0
                session_elapsed_min = (time.time() * 1000 - started_at) / 60000.0
                task_phase = row[1]
                test_failures = row[2] or 0

        return {
            "session_elapsed_min": max(session_elapsed_min, 0.0),
            "task_phase": task_phase,
            "test_failures": test_failures,
        }

    def _write(
        self,
        conn: sqlite3.Connection,
        model: str,
        result: dict,
        confidence: float,
        ttl_sec: int | None,
    ) -> None:
        now_ms = int(time.time() * 1000)
        expires_ms = (now_ms + ttl_sec * 1000) if ttl_sec else None
        conn.execute(
            "INSERT INTO ml_predictions (model, result, confidence, created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
            (model, json.dumps(result), round(confidence, 4), now_ms, expires_ms),
        )

    def _active_task_id(self, conn: sqlite3.Connection) -> str | None:
        row = conn.execute(
            "SELECT id FROM tasks WHERE phase != 'idle' AND completed_at IS NULL ORDER BY last_active DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None

    def _quality_features(self, conn: sqlite3.Connection) -> dict:
        now_ms = int(time.time() * 1000)
        window_start = now_ms - QUALITY_WINDOW_SEC * 1000
        window = [e for e in self._buffer if e.get("ts", 0) >= window_start]

        edit_events = [e for e in window if e.get("kind") == "file"]
        edits = len(edit_events)
        files: set[str] = set()
        for e in edit_events:
            p = e.get("payload") or {}
            if isinstance(p, dict) and "path" in p:
                files.add(p["path"])
        edit_focus = 1.0 - (len(files) / max(edits, 1))

        commit_events = [e for e in window if e.get("kind") == "git"]
        terminal_events = [e for e in window if e.get("kind") == "terminal"]

        row = conn.execute(
            "SELECT test_runs, test_fails, commit_count FROM tasks "
            "WHERE completed_at IS NOT NULL ORDER BY completed_at DESC LIMIT 1"
        ).fetchone()

        if row and row[0]:
            test_pass_rate = 1.0 - (row[1] / max(row[0], 1))
            baseline_commits = max(row[2], 1)
        else:
            test_pass_rate = 0.7
            baseline_commits = 1

        return {
            "test_pass_rate": max(0.0, min(1.0, test_pass_rate)),
            "test_total": row[0] if row else 0,
            "edit_focus": max(0.0, min(1.0, edit_focus)),
            "velocity_ratio": min(edits / max(len(terminal_events), 1), 2.0),
            "commits_in_window": len(commit_events),
            "expected_commits": baseline_commits / 8.0,
            "revert_count": 0,
            "edits_in_window": edits,
        }

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn
