# Python 3.13: matches the host toolchain and has universal wheel coverage for
# pydantic/psycopg/pyarrow. Bump to 3.14 here if ever needed - nothing else changes.
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install deps first so source edits don't invalidate the dependency layer.
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --no-cache -e ".[notebook,dev]"

CMD ["python", "-m", "pipeline", "--help"]
