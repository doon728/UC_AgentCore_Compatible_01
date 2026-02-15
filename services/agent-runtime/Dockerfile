FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy only dependency files first (better layer caching)
COPY pyproject.toml poetry.lock* /app/

# Install deps (no project install needed)
RUN poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi --no-root

# Copy source
COPY src/ /app/src
COPY tests/ /app/tests

# Default command (simple; replace later with your real runner)
CMD ["python", "-c", "from src.graph.build_graph import run_graph; print(run_graph('hello'))"]
