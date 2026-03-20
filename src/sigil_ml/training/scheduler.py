"""Training scheduler — fires retraining when enough new data has accumulated."""
import logging
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)

MIN_NEW_TASKS = 10          # retrain after 10 new completed tasks
MIN_INTERVAL_SEC = 3600     # no more than once per hour


class TrainingScheduler:
    """Monitors completed task count and triggers retraining when due.

    Args:
        db_path: Path to sigild SQLite database.
        reload_callback: Called after successful retraining to reload
                         model instances into the running poller.
    """

    def __init__(self, db_path: Path, reload_callback) -> None:
        self.db_path = db_path
        self._reload = reload_callback
        self._last_retrain: float = 0.0
        self._baseline_tasks: int = self._count_completed()

    def check_and_retrain(self) -> None:
        """Call from a background thread — blocks while training runs."""
        elapsed = time.time() - self._last_retrain
        if elapsed < MIN_INTERVAL_SEC and self._last_retrain > 0:
            return

        current = self._count_completed()
        if (current - self._baseline_tasks) < MIN_NEW_TASKS:
            return

        logger.info(
            "scheduler: triggering retrain (%d new tasks)",
            current - self._baseline_tasks,
        )
        from sigil_ml.training.trainer import Trainer

        try:
            result = Trainer(self.db_path).train_all()
            self._last_retrain = time.time()
            self._baseline_tasks = current
            self._log_retrain(result)
            self._reload()
            logger.info("scheduler: retrain complete — %s", result)
        except Exception:
            logger.exception("scheduler: retrain failed")

    def _count_completed(self) -> int:
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            conn.execute("PRAGMA busy_timeout=5000")
            try:
                row = conn.execute(
                    "SELECT COUNT(*) FROM tasks WHERE completed_at IS NOT NULL"
                ).fetchone()
                return row[0] if row else 0
            finally:
                conn.close()
        except sqlite3.OperationalError:
            return 0

    def _log_retrain(self, result: dict) -> None:
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            conn.execute("PRAGMA busy_timeout=5000")
            try:
                conn.execute(
                    "INSERT INTO ml_events (kind, endpoint, routing, latency_ms, ts) "
                    "VALUES ('retrain', 'scheduler', 'local', ?, ?)",
                    (
                        int(result.get("duration_sec", 0) * 1000),
                        int(time.time() * 1000),
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        except sqlite3.OperationalError:
            logger.warning("scheduler: failed to log retrain event")
