---
work_package_id: WP01
title: ServingMode Enum, Config, and CLI Flag
lane: planned
dependencies: []
subtasks:
- T001
- T002
- T003
- T004
- T005
phase: Phase 1 - Foundation
assignee: ''
agent: ''
shell_pid: ''
review_status: ''
reviewed_by: ''
history:
- timestamp: '2026-03-29T16:29:58Z'
  lane: planned
  agent: system
  shell_pid: ''
  action: Prompt generated via /spec-kitty.tasks
requirement_refs:
- FR-001
- FR-002
- FR-006
- FR-008
---

# Work Package Prompt: WP01 -- ServingMode Enum, Config, and CLI Flag

## Review Feedback Status

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

## Objectives & Success Criteria

- Introduce a `ServingMode` enum (`local` / `cloud`) that the rest of the codebase can import and branch on.
- Wire `--mode local|cloud` into the CLI `serve` subcommand.
- Make `create_app()` accept the mode and conditionally skip SQLite, the EventPoller, and the TrainingScheduler when running in cloud mode.
- Existing `sigil-ml serve` (no flag) must behave identically to today -- zero regressions.
- The module-level `app = create_app()` line must continue to work for `uvicorn sigil_ml.app:app`.

**Success gate**: `sigil-ml serve --mode cloud` starts without opening SQLite, without starting a poller, and responds on its configured port. `sigil-ml serve` (default) still starts poller + SQLite as before.

## Context & Constraints

- **Spec**: `kitty-specs/001-cloud-serving-mode/spec.md` -- FR-001, FR-002, FR-006, FR-008.
- **CLAUDE.md invariant**: No heavyweight dependencies beyond `scikit-learn`, `numpy`, `fastapi`, `uvicorn`, `joblib`.
- **Current architecture**: `app.py` has a single `create_app()` that unconditionally bootstraps SQLite schema, loads models from local disk, starts the EventPoller, and starts a TrainingScheduler loop.
- **Backward compatibility**: The line `app = create_app()` at module level is used by `uvicorn sigil_ml.app:app`. It MUST default to local mode.

## Subtasks & Detailed Guidance

### Subtask T001 -- Create `ServingMode` enum in `src/sigil_ml/config.py`

- **Purpose**: Provide a single source of truth for the operating mode that all modules can import.
- **Steps**:
  1. Add an `enum.Enum` (or `str` enum for JSON-friendliness) to `src/sigil_ml/config.py`:
     ```python
     import enum

     class ServingMode(str, enum.Enum):
         LOCAL = "local"
         CLOUD = "cloud"
     ```
  2. Add a module-level function to resolve the active mode:
     ```python
     def serving_mode(override: str | None = None) -> ServingMode:
         """Resolve the active serving mode.

         Priority: explicit override > env var > default (local).
         """
         raw = override or os.environ.get("SIGIL_ML_MODE", "local")
         try:
             return ServingMode(raw.lower())
         except ValueError:
             raise SystemExit(f"Invalid serving mode: {raw!r}. Must be 'local' or 'cloud'.")
     ```
  3. Keep the existing `db_path()`, `models_dir()`, `weights_path()`, and `sigild_plugin_url()` functions unchanged.
- **Files**: `src/sigil_ml/config.py`
- **Parallel?**: No -- downstream subtasks depend on this.
- **Notes**: Using `str, enum.Enum` ensures the value serializes cleanly to JSON (for health endpoints later). The `serving_mode()` function consolidates override logic so callers don't have to.

### Subtask T002 -- Add `--mode` argument to the `serve` subcommand in `src/sigil_ml/cli.py`

- **Purpose**: Let operators choose the serving mode from the CLI.
- **Steps**:
  1. In `cli.py`, add a `--mode` argument to the `serve` subparser:
     ```python
     serve_parser.add_argument(
         "--mode",
         choices=["local", "cloud"],
         default=None,
         help="Serving mode: 'local' (default, with poller) or 'cloud' (stateless K8s)",
     )
     ```
  2. When `args.command == "serve"`, resolve the mode and pass it to the app:
     ```python
     from sigil_ml.config import serving_mode

     mode = serving_mode(args.mode)
     ```
  3. Store the resolved mode in an environment variable so that `create_app()` (invoked by uvicorn in a separate import) can pick it up:
     ```python
     import os
     os.environ["_SIGIL_ML_MODE_OVERRIDE"] = mode.value
     ```
  4. Then call `uvicorn.run(...)` as before.
- **Files**: `src/sigil_ml/cli.py`
- **Parallel?**: No -- depends on T001.
- **Notes**: We use an env var bridge because uvicorn imports `sigil_ml.app:app` as a fresh module; we cannot pass arguments directly. The env var `_SIGIL_ML_MODE_OVERRIDE` is internal (underscored) to avoid collision with the public `SIGIL_ML_MODE`.

### Subtask T003 -- Thread `ServingMode` into `create_app()` via a factory parameter in `src/sigil_ml/app.py`

- **Purpose**: Make `create_app()` mode-aware so it can conditionally initialize components.
- **Steps**:
  1. Update the `create_app` signature:
     ```python
     from sigil_ml.config import ServingMode, serving_mode

     def create_app(mode: ServingMode | None = None) -> FastAPI:
         if mode is None:
             # Check for CLI bridge env var, then fall back to SIGIL_ML_MODE, then default
             override = os.environ.pop("_SIGIL_ML_MODE_OVERRIDE", None)
             mode = serving_mode(override)
         ...
     ```
  2. Store the mode on `AppState`:
     ```python
     class AppState:
         def __init__(self, mode: ServingMode = ServingMode.LOCAL) -> None:
             self.mode = mode
             # ... existing fields ...
     ```
  3. Pass mode into `AppState`:
     ```python
     state = AppState(mode=mode)
     ```
  4. The module-level `app = create_app()` continues to work -- it will resolve to local mode by default.
- **Files**: `src/sigil_ml/app.py`
- **Parallel?**: No -- T004 depends on this.
- **Notes**: Using `os.environ.pop()` cleans up the bridge var after reading it, preventing leakage.

### Subtask T004 -- Conditional startup: skip SQLite, poller, and training scheduler in cloud mode

- **Purpose**: In cloud mode, the service must start without local dependencies. FR-002 mandates no SQLite connections; FR-008 mandates no writes to SQLite tables.
- **Steps**:
  1. In the `startup_event()` inside `create_app()`, wrap the SQLite + poller + scheduler block in a mode check:
     ```python
     @application.on_event("startup")
     async def startup_event() -> None:
         if state.mode == ServingMode.LOCAL:
             db = config.db_path()
             try:
                 ensure_ml_tables(db)
             except Exception:
                 logger.warning("schema bootstrap failed", exc_info=True)

             state.load_models()

             state.poller = EventPoller(
                 db_path=db,
                 models={ ... },
             )
             asyncio.create_task(state.poller.run())

             scheduler = TrainingScheduler(db, reload_callback=state.reload_models_into_poller)
             asyncio.create_task(_schedule_loop(scheduler))

             logger.info("sigil-ml: local mode — models loaded, poller started, scheduler active")
         else:
             # Cloud mode: no SQLite, no poller, no scheduler
             # Models will be loaded on-demand per tenant (WP05)
             logger.info("sigil-ml: cloud mode — stateless serving, no poller")
     ```
  2. Update `shutdown_event()` similarly:
     ```python
     @application.on_event("shutdown")
     async def shutdown_event() -> None:
         if state.poller:
             state.poller.stop()
             logger.info("poller stopped")
     ```
     (This already handles cloud mode since `state.poller` will be `None`.)
  3. In cloud mode, `state.load_models()` is NOT called at startup. Models are loaded on-demand per tenant in WP05. For now (before WP05), cloud mode will serve fallback/rule-based predictions for all endpoints.
- **Files**: `src/sigil_ml/app.py`
- **Parallel?**: Yes (independent of T005 once T003 is done).
- **Notes**: After this subtask, cloud mode starts but all predict endpoints will return fallback responses (models are None). This is correct -- WP05 adds tenant-aware model loading.

### Subtask T005 -- Add environment variable override `SIGIL_ML_MODE`

- **Purpose**: Allow operators to set the mode via environment variable (useful in Kubernetes ConfigMaps/Secrets) without requiring CLI flags.
- **Steps**:
  1. This is already handled in T001's `serving_mode()` function which reads `SIGIL_ML_MODE`.
  2. Verify the priority chain works: `--mode` flag > `SIGIL_ML_MODE` env var > default `"local"`.
  3. Document the env var in a docstring or comment near `serving_mode()`.
  4. Add a brief log line at startup showing the resolved mode and its source:
     ```python
     logger.info("sigil-ml: resolved mode=%s (source=%s)", mode.value, source)
     ```
     where source is `"cli"`, `"env"`, or `"default"`.
- **Files**: `src/sigil_ml/config.py`, `src/sigil_ml/app.py`
- **Parallel?**: Yes (independent of T004).
- **Notes**: The `serving_mode()` function from T001 already reads the env var. This subtask is about ensuring the priority chain is correct and observable.

## Risks & Mitigations

- **Breaking `uvicorn sigil_ml.app:app`**: The module-level `app = create_app()` must default to local mode. Mitigated by making `mode` parameter optional with local default.
- **Env var collision**: `_SIGIL_ML_MODE_OVERRIDE` is internal (underscored prefix). `SIGIL_ML_MODE` is the public env var.
- **Cloud mode with no models**: After WP01, cloud mode starts but all endpoints return fallback responses. This is intentional -- WP02 makes endpoints stateless, WP05 adds model loading.

## Review Guidance

- Verify `sigil-ml serve` (no flags) still starts poller + SQLite as before.
- Verify `sigil-ml serve --mode cloud` starts without any SQLite operations in logs.
- Verify `SIGIL_ML_MODE=cloud sigil-ml serve` works the same as `--mode cloud`.
- Verify `app = create_app()` at module level defaults to local mode.
- Check that no SQLite imports are executed at module scope (they should only be in functions guarded by mode checks).

## Activity Log

- 2026-03-29T16:29:58Z -- system -- lane=planned -- Prompt created.
