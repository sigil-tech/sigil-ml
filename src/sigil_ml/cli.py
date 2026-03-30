"""CLI entry point for sigil-ml."""

import argparse
import os
import sys

import uvicorn

from sigil_ml.config import resolve_mode
from sigil_ml.storage.model_store import model_store_factory
from sigil_ml.store import create_store
from sigil_ml.store_sqlite import SqliteStore
from sigil_ml.training.trainer import Trainer


def main() -> None:
    """Entry point for the sigil-ml CLI."""
    parser = argparse.ArgumentParser(description="Sigil ML sidecar")
    sub = parser.add_subparsers(dest="command")

    serve_parser = sub.add_parser("serve", help="Start the ML server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=7774)
    serve_parser.add_argument(
        "--mode",
        choices=["local", "cloud"],
        default=None,
        help="Serving mode: 'local' (default, with poller) or 'cloud' (stateless, no SQLite)",
    )

    train_parser = sub.add_parser("train", help="Train models from local data")
    train_parser.add_argument("--db", help="Path to sigild SQLite database")

    sub.add_parser("health-check", help="Check if server is running")

    args = parser.parse_args()

    if args.command == "serve":
        mode = resolve_mode(args.mode)
        # Bridge mode to create_app() via env var (uvicorn string import cannot pass args)
        os.environ["SIGIL_ML_MODE"] = mode.value
        uvicorn.run(
            "sigil_ml.app:app",
            host=args.host,
            port=args.port,
            log_level="info",
        )
    elif args.command == "train":
        if args.db:
            from pathlib import Path

            store = SqliteStore(Path(args.db))
        else:
            store = create_store()
        ms = model_store_factory()
        print(f"Training models using {type(store).__name__} + {type(ms).__name__} ...")
        trainer = Trainer(store, model_store=ms)
        result = trainer.train_all()
        print(f"Done: {result}")
    elif args.command == "health-check":
        try:
            import httpx
        except ImportError:
            print("httpx is required for health-check: pip install httpx", file=sys.stderr)
            sys.exit(1)

        try:
            resp = httpx.get("http://127.0.0.1:7774/health", timeout=5)
            data = resp.json()
            print(f"Status: {data['status']}")
            for model, state in data.get("models", {}).items():
                print(f"  {model}: {state}")
        except Exception as e:
            print(f"Server not reachable: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
