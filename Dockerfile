# ======================================================================
# Stage 1: Build React + Vite frontend
# ======================================================================
FROM node:20-slim as frontend-builder

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ======================================================================
# Stage 2: Build Python dependencies & compile C/Cython extensions
# ======================================================================
FROM python:3.11-slim as backend-builder

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

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade packaging tools and install Cython + NumPy directly into virtualenv
RUN pip install --no-cache-dir --upgrade pip setuptools wheel cython numpy==1.26.4

# Install all requirements and web dependencies directly into virtualenv
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn websockets python-multipart httpx

# ======================================================================
# Stage 3: Final minimal production runtime
# ======================================================================
FROM python:3.11-slim as runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000

WORKDIR /app

# Install only the runtime shared libraries needed by OpenCV, ONNXRuntime, and InsightFace
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the complete virtual environment from backend-builder
COPY --from=backend-builder /opt/venv /opt/venv

# Copy compiled static frontend from frontend-builder stage
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Copy application source, configuration, and data directories
COPY app/ ./app/
COPY config.yaml ./config.yaml
COPY data/ ./data/

# Expose port (Render overrides this dynamically via $PORT)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Launch ASGI server with dynamic $PORT evaluation
CMD uvicorn app.web.server:app --host 0.0.0.0 --port ${PORT}
