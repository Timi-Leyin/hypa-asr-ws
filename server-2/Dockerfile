# ─────────────────────────────────────────────────────────────────────────────
# WebSocket Transcription Server - Docker Image
# Optimized for NVIDIA H200 / H100 / A100 GPUs with CUDA 12.x
# ─────────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir \
    faster-whisper>=1.0.0 \
    websockets>=12.0 \
    numpy>=1.26.0 \
    librosa>=0.10.0 \
    transformers>=4.57.1 \
    accelerate>=0.30.0

# Copy application code
COPY websocket_server.py .
COPY wspr_small_ct2/ ./wspr_small_ct2/

# Expose WebSocket port
EXPOSE 8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.connect(('localhost',8765)); s.close()" || exit 1

# Run the server
CMD ["python", "websocket_server.py"]
