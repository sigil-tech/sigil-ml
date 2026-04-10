"""Feature store integration — bridges sigil-ml extractors with Feast online store.

Materialization: compute features via existing extractors -> push to Feast online store.
Retrieval: fetch features from Feast online store by entity ID.

This module is optional — everything works without it (graceful fallback).
Install feast[sqlite] to enable local online store serving.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class SigilFeatureStore:
    """Wraps Feast to provide feature materialization and retrieval.

    Materialization: compute features via existing extractors and push to the
    Feast online store so future requests can retrieve them by entity ID.

    Retrieval: fetch features from the Feast online store by task_id, eliminating
    the need for API callers to compute and send features in request bodies.
    """

    def __init__(self, repo_path: str | None = None) -> None:
        from feast import FeatureStore

        if repo_path is None:
            import importlib.resources

            repo_path = str(importlib.resources.files("sigil_ml") / "feast_repo")

        self._store = FeatureStore(repo_path=repo_path)
        logger.info("Feast feature store initialized from %s", repo_path)

    def apply(self) -> None:
        """Register feature views and entities with the Feast registry.

        Call once at startup (or via 'make feast-apply') to synchronize the
        feature_store.yaml definitions with the registry database.
        """
        from sigil_ml.feast_repo.features import (
            duration_features,
            duration_push_source,
            fleet_focus_features,
            fleet_focus_push_source,
            node_entity,
            stuck_features,
            stuck_push_source,
            task_entity,
        )

        self._store.apply(
            [
                task_entity,
                node_entity,
                stuck_push_source,
                duration_push_source,
                fleet_focus_push_source,
                stuck_features,
                duration_features,
                fleet_focus_features,
            ]
        )
        logger.info("Feast feature definitions applied to registry")

    def materialize_task(self, data_store: Any, task_id: str) -> None:
        """Compute and push features for a task into the online store.

        Uses the existing extractors from sigil_ml.features to compute feature
        values, then pushes them into the Feast online store keyed by task_id.

        Args:
            data_store: A DataStore instance to query task/event data from.
            task_id: The task identifier to materialize features for.
        """
        from sigil_ml.features import extract_duration_features, extract_stuck_features

        stuck = extract_stuck_features(data_store, task_id)
        duration = extract_duration_features(data_store, task_id)

        now = datetime.now(timezone.utc)

        self._store.push(
            push_source_name="stuck_push_source",
            df=_to_df({"task_id": task_id, **stuck}, now),
        )
        self._store.push(
            push_source_name="duration_push_source",
            df=_to_df({"task_id": task_id, **duration}, now),
        )
        logger.debug("Materialized features for task %s", task_id)

    def get_stuck_features(self, task_id: str) -> dict[str, float]:
        """Retrieve stuck features from the online store by task_id.

        Args:
            task_id: The task identifier to look up.

        Returns:
            Dict of feature name -> value. May contain None values if the
            entity has not been materialized yet.
        """
        response = self._store.get_online_features(
            features=[
                "stuck_features:test_failure_count",
                "stuck_features:time_in_phase_sec",
                "stuck_features:edit_velocity",
                "stuck_features:file_switch_rate",
                "stuck_features:session_length_sec",
                "stuck_features:time_since_last_commit_sec",
            ],
            entity_rows=[{"task_id": task_id}],
        ).to_dict()

        # With full_feature_names=False (default) keys are plain feature names.
        # Exclude entity key(s) so only numeric features are returned.
        return {k: v[0] for k, v in response.items() if k != "task_id"}

    def get_duration_features(self, task_id: str) -> dict[str, float]:
        """Retrieve duration features from the online store by task_id.

        Args:
            task_id: The task identifier to look up.

        Returns:
            Dict of feature name -> value. May contain None values if the
            entity has not been materialized yet.
        """
        response = self._store.get_online_features(
            features=[
                "duration_features:file_count",
                "duration_features:total_edits",
                "duration_features:time_of_day_hour",
                "duration_features:branch_name_length",
            ],
            entity_rows=[{"task_id": task_id}],
        ).to_dict()

        return {k: v[0] for k, v in response.items() if k != "task_id"}

    def has_features(self, feature_view: str, entity_id: str) -> bool:
        """Check if features exist in the online store for an entity.

        Args:
            feature_view: Name of the feature view to probe (e.g. "stuck_features").
            entity_id: The entity ID (task_id) to check.

        Returns:
            True if at least one feature value is non-null in the online store.
        """
        # Probe one representative field per feature view
        probe_fields: dict[str, str] = {
            "stuck_features": "stuck_features:test_failure_count",
            "duration_features": "duration_features:file_count",
            "fleet_focus_features": "fleet_focus_features:focus_score",
        }
        probe = probe_fields.get(feature_view, f"{feature_view}:test_failure_count")

        try:
            response = self._store.get_online_features(
                features=[probe],
                entity_rows=[{"task_id": entity_id}],
            ).to_dict()
            # With full_feature_names=False, the feature key is the plain field name
            feature_key = next((k for k in response if k != "task_id"), None)
            if feature_key is None:
                return False
            return response[feature_key][0] is not None
        except Exception:
            return False


def _to_df(row: dict[str, Any], timestamp: datetime):
    """Convert a single feature row dict to a pandas DataFrame for Feast push.

    Adds an event_timestamp column required by Feast for time-travel semantics.

    Args:
        row: Dict of feature values including the entity join key (e.g. task_id).
        timestamp: The event timestamp to attach to this feature row.

    Returns:
        A pandas DataFrame with one row and an event_timestamp column.
    """
    import pandas as pd

    df = pd.DataFrame([row])
    df["event_timestamp"] = timestamp
    return df
