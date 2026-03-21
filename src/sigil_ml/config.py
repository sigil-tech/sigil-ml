"""Configuration and path discovery for sigil-ml."""

import os
from pathlib import Path


def _data_home() -> Path:
    """Return the XDG data home, defaulting to ~/.local/share."""
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))


def db_path() -> Path:
    """Return the path to the sigild SQLite database."""
    return _data_home() / "sigild" / "data.db"


def models_dir() -> Path:
    """Return the directory for ML model weights, creating it if needed."""
    d = _data_home() / "sigild" / "ml-models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def weights_path(model_name: str) -> Path:
    """Return the path to a specific model's weight file."""
    return models_dir() / f"{model_name}.joblib"


def sigild_plugin_url() -> str:
    """Return the URL for the sigild plugin ingest/capabilities API."""
    return os.environ.get("SIGILD_PLUGIN_URL", "http://127.0.0.1:7775")
