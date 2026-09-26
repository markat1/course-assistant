FROM python:3.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.12 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY gateway/ ./gateway/

CMD ["/app/.venv/bin/uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8780"]