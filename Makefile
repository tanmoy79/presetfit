.PHONY: install lint format typecheck test check serve docker frontend frontend-check frontend-dev

install:  ## Install the project with development and training tools, and build the web page
	pip install -e ".[dev,train,bench]"
	cd frontend && npm install && npm run build

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

check: lint typecheck test frontend-check  ## Everything CI runs

serve:  ## Start the web app at http://127.0.0.1:8000
	presetfit serve

docker:  ## Build the container image
	docker build -t presetfit .

frontend:  ## Build the web page into the Python package
	cd frontend && npm run build

frontend-check:  ## Lint, type-check and test the web page
	cd frontend && npm run lint && npm run typecheck && npm test

frontend-dev:  ## Page with live reload at http://localhost:5173 (needs `presetfit serve` running)
	cd frontend && npm run dev
