# Multi-stage container build for VisionGuard AI Web Demo
FROM python:3.11-slim as base

# Prevent Python from buffering stdout/stderr and writing pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system runtime dependencies for OpenCV and image decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn websockets

# Copy application files
COPY app/ ./app/
COPY config.yaml ./config.yaml
COPY data/ ./data/

# Expose web port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

# Launch ASGI server
CMD ["uvicorn", "app.web.server:app", "--host", "0.0.0.0", "--port", "8000"]
