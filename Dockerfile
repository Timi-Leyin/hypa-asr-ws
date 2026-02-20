# FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
# ENV DEBIAN_FRONTEND=noninteractive
# ENV PYTHONUNBUFFERED=1
# ENV PYTHONDONTWRITEBYTECODE=1

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3.11 \
#     python3.11-venv \
#     python3.11-dev \
#     python3-pip \
#     ffmpeg \
#     libsndfile1 \
#     git \
#     && rm -rf /var/lib/apt/lists/* \
#     && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
#     && ln -sf /usr/bin/python3.11 /usr/bin/python

# WORKDIR /app

# COPY requirements.txt .

# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir \
#     torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124 && \
#     pip install --no-cache-dir \
#     faster-whisper>=1.0.0 \
#     websockets>=12.0 \
#     numpy>=1.26.0 \
#     librosa>=0.10.0 \
#     transformers>=4.57.1 \
#     accelerate>=0.30.0

    
# COPY websocket_server.py .
# COPY wspr_small_ct2/ ./wspr_small_ct2/

# EXPOSE 8765

# HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
#     CMD python -c "import asyncio, websockets; async def m():\n  async with websockets.connect('ws://localhost:8765'):\n    return\nasyncio.run(m())" || exit 1

# CMD ["python", "websocket_server.py"]


FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD [ "python", "-u", "handler.py" ]