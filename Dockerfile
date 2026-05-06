# Multi-stage Dockerfile for the rankmle web app.
#
# Stage 1: download a precompiled KataGo binary (Eigen / CPU). For GPU
# deployments, swap KATAGO_URL for the cuda or opencl variant.
#
# Stage 2: slim Python runtime with uv-managed deps, the binary, and the app.
# The human SL model is NOT baked in — mount it at /models/human.bin.gz or
# override KATAGO_HUMAN_MODEL. A sensible analysis config is built in; mount
# /config/analysis.cfg and set KATAGO_CONFIG to override it.

ARG KATAGO_VERSION=1.15.3
ARG KATAGO_URL=https://github.com/lightvector/KataGo/releases/download/v${KATAGO_VERSION}/katago-v${KATAGO_VERSION}-eigen-linux-x64.zip

FROM debian:bookworm-slim AS katago
ARG KATAGO_URL
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates unzip \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /tmp
RUN curl -fL "$KATAGO_URL" -o katago.zip \
    && unzip katago.zip -d katago \
    && find katago -name katago -type f -executable -exec cp {} /usr/local/bin/katago \; \
    && find katago -name '*.cfg' -exec cp {} /usr/local/share/ \;

FROM python:3.12-slim AS runtime
COPY --from=katago /usr/local/bin/katago /usr/local/bin/katago
COPY --from=katago /usr/local/share/*.cfg /usr/local/share/

# Eigen binary needs libgomp at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv for dependency management
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH

# Install deps first for better layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Then copy the app
COPY *.py ./

ENV KATAGO_BIN=/usr/local/bin/katago \
    KATAGO_HUMAN_MODEL=/models/human.bin.gz \
    KATAGO_MODEL=/models/kata1.bin.gz \
    UPLOAD_DIR=/data/uploads

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
