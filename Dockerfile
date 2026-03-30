# syntax=docker/dockerfile:1

# ──────────────────────────────────────────────
# Builder stage
# ──────────────────────────────────────────────
FROM python:3.12 AS builder

LABEL maintainer="RosettaStone <noreply@rosettastone>" \
      description="RosettaStone — automated LLM model migration tool"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /build

# Copy only the files needed for dependency installation first (layer caching)
COPY pyproject.toml .
COPY src/ src/

# Install all dependencies (including the [all] extras) into the system Python
RUN uv pip install --system ".[all]"

# ──────────────────────────────────────────────
# Runtime stage
# ──────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="RosettaStone <noreply@rosettastone>" \
      description="RosettaStone — automated LLM model migration tool"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install curl (required for HEALTHCHECK)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN groupadd --system rosettastone \
    && useradd --system --gid rosettastone --no-create-home rosettastone

# Copy installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code
WORKDIR /app
COPY src/ src/

# Transfer ownership to the non-root user
RUN chown -R rosettastone:rosettastone /app

USER rosettastone

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "rosettastone.server.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
