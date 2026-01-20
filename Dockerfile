# syntax=docker/dockerfile:1

# ============================================================
# Production Dockerfile for Leads Agent
# ============================================================

# --- Build stage ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy dependency files first for better layer caching
COPY pyproject.toml ./
COPY README.md ./
COPY LICENSE ./

# Copy source code
COPY src/ ./src/

# Install dependencies and the package (NOT editable for production)
RUN uv pip install --system --no-cache .


# --- Runtime stage ---
FROM python:3.11-slim AS runtime

WORKDIR /app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy installed packages from builder (includes leads_agent in site-packages)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/leads-agent /usr/local/bin/leads-agent
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Switch to non-root user
USER appuser

# Environment defaults (override at runtime)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Expose the default port
EXPOSE 8000

# Health check (API has health endpoint at /)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Run the API server
CMD ["leads-agent", "run", "--host", "0.0.0.0", "--port", "8000"]
