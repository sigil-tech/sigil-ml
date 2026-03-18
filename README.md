# sigil-ml

ML sidecar for the [Sigil](https://github.com/alecfeeman/sigil) developer intelligence daemon.

Provides real-time predictions to help developers stay productive:

- **Stuck detection** -- predicts when you're stuck on a task using edit patterns, test failures, and timing signals
- **Suggestion policy** -- Thompson Sampling bandit that learns which nudges (commit, step back, take a break) actually help you
- **Duration estimation** -- estimates how long a task will take based on file count, edit volume, and historical data

## Install

```bash
pip install -e ".[dev]"
```

## Usage

### Start the server

```bash
sigil-ml serve --port 7774
```

### Train models

From local sigild data:

```bash
sigil-ml train
```

Or point at a specific database:

```bash
sigil-ml train --db /path/to/data.db
```

### Health check

```bash
sigil-ml health-check
```

### API endpoints

| Endpoint             | Method | Description                        |
|----------------------|--------|------------------------------------|
| `/health`            | GET    | Model readiness and uptime         |
| `/predict/stuck`     | POST   | Stuck probability for a task       |
| `/predict/suggest`   | POST   | Next best suggestion action        |
| `/predict/duration`  | POST   | Estimated task duration in minutes |
| `/train`             | POST   | Trigger background retraining      |

## Run tests

```bash
pytest tests/
```

## Architecture

```
sigil-ml/
  src/sigil_ml/
    config.py          # Path discovery (XDG-aware)
    features.py        # Feature extraction from SQLite
    server.py          # FastAPI server + CLI
    models/
      stuck.py         # GradientBoostingClassifier
      suggest.py       # Thompson Sampling bandit
      duration.py      # GradientBoostingRegressor
    training/
      synthetic.py     # Synthetic data generation
      trainer.py       # Orchestrated retraining
```

## License

Apache-2.0
