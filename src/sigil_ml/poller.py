"""Event poller — continuous push-to-db prediction loop.

Polls events → runs models → writes to ml_predictions.
Runs as an asyncio background task inside the FastAPI process.
"""
import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path

from sigil_ml.features import (
    extract_stuck_features,
    extract_suggest_features,
    extract_duration_features,
    extract_features_from_buffer,
)

logger = logging.getLogger(__name__)

POLL_INTERVAL_SEC = 0.5
PREDICT_EVERY_N_EVENTS = 3
QUALITY_WINDOW_SEC = 1800   # 30-minute rolling window
PREDICTION_TTL_SEC = 300    # 5-minute expiry for stuck/suggest
QUALITY_TTL_SEC = 1800      # 30-minute expiry for quality


class EventPoller:
    """Polls sigild's events table and writes predictions to ml_predictions."""

    def __init__(self, db_path: Path, models: dict) -> None:
        self.db_path = db_path
        self.stuck = models["stuck"]
        self.suggest = models["suggest"]
        self.duration = models["duration"]
        self.quality = models["quality"]
        self._buffer: list[dict] = []
        self._since_last_predict = 0
        self._running = False

    async def run(self) -> None:
        """Main loop — call as an asyncio task."""
        self._running = True
        logger.info("poller: started against %s", self.db_path)
        while self._running:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, self._poll_once
                )
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
            cursor_id = conn.execute(
                "SELECT last_event_id FROM ml_cursor WHERE id = 1"
            ).fetchone()
            since = cursor_id[0] if cursor_id else 0

            rows = conn.execute(
                "SELECT id, kind, source, payload, ts FROM events "
                "WHERE id > ? ORDER BY id ASC LIMIT 100",
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
                events.append(e)

            self._buffer.extend(events)
            self._buffer = self._buffer[-50:]  # keep last 50
            self._since_last_predict += len(events)

            max_id = max(e["id"] for e in events)
            conn.execute(
                "UPDATE ml_cursor SET last_event_id = ?, updated_at = ? WHERE id = 1",
                (max_id, int(time.time() * 1000)),
            )

            if self._since_last_predict >= PREDICT_EVERY_N_EVENTS:
                self._predict_and_write(conn)
                self._since_last_predict = 0

            conn.commit()
        finally:
            conn.close()

    def _predict_and_write(self, conn: sqlite3.Connection) -> None:
        start = time.time()
        task_id = self._active_task_id(conn)

        # Stuck prediction
        if task_id:
            feats = extract_stuck_features(self.db_path, task_id)
        else:
            feats = extract_features_from_buffer(self._buffer)
        result = self.stuck.predict(feats)
        self._write(conn, "stuck", result, result.get("probability", 0.0), PREDICTION_TTL_SEC)

        # Suggestion policy
        state = extract_suggest_features(self.db_path, task_id) if task_id else None
        result = self.suggest.predict(state)
        self._write(conn, "suggest", result, result.get("confidence", 0.0), PREDICTION_TTL_SEC)

        # Duration (only when there is an active task)
        if task_id:
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

        # Quality score
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
            "INSERT INTO ml_predictions (model, result, confidence, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (model, json.dumps(result), round(confidence, 4), now_ms, expires_ms),
        )

    def _active_task_id(self, conn: sqlite3.Connection) -> str | None:
        row = conn.execute(
            "SELECT id FROM tasks WHERE phase != 'idle' AND completed_at IS NULL "
            "ORDER BY last_active DESC LIMIT 1"
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
