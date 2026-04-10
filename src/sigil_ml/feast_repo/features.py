"""Feast feature definitions for sigil-ml.

Entities and feature views for each ML model. Features are pushed into the
online store via PushSource — computation is handled by the existing extractors
in sigil_ml.features and materialized by SigilFeatureStore.

Note on batch_source: Feast requires PushSource to declare a batch_source.
We use placeholder FileSource entries pointing to non-existent paths because
sigil-ml never runs offline batch materialization — all features are pushed
to the online store in real time by the event poller.
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field, PushSource, ValueType
from feast.infra.offline_stores.file_source import FileSource
from feast.types import Float64, Int64

# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

task_entity = Entity(name="task", join_keys=["task_id"], value_type=ValueType.STRING)
node_entity = Entity(name="node", join_keys=["node_id"], value_type=ValueType.STRING)

# ---------------------------------------------------------------------------
# Placeholder batch sources (required by Feast; never read in practice)
# ---------------------------------------------------------------------------

_stuck_batch = FileSource(path="data/stuck_features.parquet", timestamp_field="event_timestamp")
_duration_batch = FileSource(path="data/duration_features.parquet", timestamp_field="event_timestamp")
_fleet_focus_batch = FileSource(path="data/fleet_focus_features.parquet", timestamp_field="event_timestamp")

# ---------------------------------------------------------------------------
# Push sources (features arrive via SigilFeatureStore.materialize_task())
# ---------------------------------------------------------------------------

stuck_push_source = PushSource(
    name="stuck_push_source",
    batch_source=_stuck_batch,
)

duration_push_source = PushSource(
    name="duration_push_source",
    batch_source=_duration_batch,
)

fleet_focus_push_source = PushSource(
    name="fleet_focus_push_source",
    batch_source=_fleet_focus_batch,
)

# ---------------------------------------------------------------------------
# Feature views
# ---------------------------------------------------------------------------

stuck_features = FeatureView(
    name="stuck_features",
    entities=[task_entity],
    schema=[
        Field(name="test_failure_count", dtype=Float64),
        Field(name="time_in_phase_sec", dtype=Float64),
        Field(name="edit_velocity", dtype=Float64),
        Field(name="file_switch_rate", dtype=Float64),
        Field(name="session_length_sec", dtype=Float64),
        Field(name="time_since_last_commit_sec", dtype=Float64),
    ],
    source=stuck_push_source,
    ttl=timedelta(hours=1),
)

duration_features = FeatureView(
    name="duration_features",
    entities=[task_entity],
    schema=[
        Field(name="file_count", dtype=Float64),
        Field(name="total_edits", dtype=Float64),
        Field(name="time_of_day_hour", dtype=Float64),
        Field(name="branch_name_length", dtype=Float64),
    ],
    source=duration_push_source,
    ttl=timedelta(hours=1),
)

fleet_focus_features = FeatureView(
    name="fleet_focus_features",
    entities=[node_entity],
    schema=[
        Field(name="focus_score", dtype=Float64),
        Field(name="meeting_minutes", dtype=Float64),
        Field(name="context_switches", dtype=Int64),
        Field(name="idle_minutes", dtype=Float64),
        Field(name="active_minutes", dtype=Float64),
    ],
    source=fleet_focus_push_source,
    ttl=timedelta(hours=24),
)
