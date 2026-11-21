# syntax=docker/dockerfile:1

# Stage 1: build the React page. Node is needed only here; the final image
# carries just the finished files.
FROM node:22-slim AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend ./
# vite.config.ts writes the page into ../src/presetfit/api/static
RUN mkdir -p ../src/presetfit/api && npm run build

# Stage 2: the Python app that serves the page and does the matching.
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY --from=frontend /build/src/presetfit/api/static ./src/presetfit/api/static
RUN pip install ".[api]"

# Never run a web service as root.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "presetfit.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
