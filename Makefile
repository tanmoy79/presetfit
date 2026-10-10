.PHONY: install lint format typecheck test check serve docker

install:  ## Install the project with development tools
	pip install -e ".[dev]"

lint:  ## Check style and common bugs
	ruff check .
	ruff format --check .

format:  ## Rewrite files to the project style
	ruff check --fix .
	ruff format .

typecheck:  ## Check type annotations
	mypy

test:  ## Run the tests with coverage
	pytest --cov --cov-report=term-missing

check: lint typecheck test  ## Everything CI runs

serve:  ## Start the web app at http://127.0.0.1:8000
	presetfit serve

docker:  ## Build the container image
	docker build -t presetfit .
