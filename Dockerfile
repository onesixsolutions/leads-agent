# syntax=docker/dockerfile:1

# ============================================================
# Production Dockerfile for Leads Agent (Socket Mode)
# ============================================================

FROM python:3.11-slim

WORKDIR /app

# Install uv and create non-root user
RUN pip install --no-cache-dir uv \
    && useradd --create-home --shell /bin/bash appuser

# Copy application code
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Install dependencies and the package (NOT editable for production)
RUN uv pip install --system --no-cache . \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment defaults (override at runtime via .env)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# No EXPOSE needed - Socket Mode uses outbound WebSocket only

# Health check - verify the process is running
# (Socket Mode doesn't have an HTTP endpoint to check)
HEALTHCHECK --interval=60s --timeout=10s --start-period=10s --retries=3 \
    CMD pgrep -f "leads-agent" || exit 1

# Run the bot in Socket Mode
CMD ["leads-agent", "run"]
