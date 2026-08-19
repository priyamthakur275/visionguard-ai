# ======================================================================
# Stage 1: Build stage with C/C++ compilation toolchain for InsightFace
# ======================================================================
FROM python:3.11-slim as builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install compilation toolchain required to build insightface C/Cython extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    gcc \
    g++ \
    cmake \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade packaging tools and pre-install Cython + NumPy for C extension compilation
RUN pip install --no-cache-dir --upgrade pip setuptools wheel cython numpy==1.26.4

# Install python dependencies into an isolated prefix
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt && \
    pip install --no-cache-dir --prefix=/install fastapi uvicorn websockets python-multipart httpx

# ======================================================================
# Stage 2: Final minimal production runtime
# ======================================================================
FROM python:3.11-slim as runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000

WORKDIR /app

# Install only the runtime shared libraries needed by OpenCV, ONNXRuntime, and InsightFace
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-compiled python packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source, configuration, data directories, and frontend bundle
COPY app/ ./app/
COPY config.yaml ./config.yaml
COPY data/ ./data/
COPY frontend/dist/ ./frontend/dist/

# Expose port (Render overrides this dynamically via $PORT)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Launch ASGI server with dynamic $PORT evaluation
CMD uvicorn app.web.server:app --host 0.0.0.0 --port ${PORT}
