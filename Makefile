.PHONY: install lint format test build clean feast-apply

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

test:
	pytest tests/ -v

# Register Feast feature views in the local registry
feast-apply:
	cd src/sigil_ml/feast_repo && feast apply

build:
	python -m build

clean:
	rm -rf dist/ build/ *.egg-info src/*.egg-info
	rm -f src/sigil_ml/feast_repo/data/registry.db
	rm -f src/sigil_ml/feast_repo/data/online_store.db
