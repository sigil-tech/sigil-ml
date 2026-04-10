.PHONY: openapi openapi-check install lint format test build clean

# Generate the OpenAPI spec from the FastAPI app
openapi:
	python scripts/gen_openapi.py

# Verify the committed spec matches the code (used by CI)
openapi-check:
	python scripts/gen_openapi.py --check

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

test:
	pytest tests/ -v

build:
	python -m build

clean:
	rm -rf dist/ build/ *.egg-info src/*.egg-info
