python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt

python main.py --audio /absolute/path/to/audio.wav

python main_faster.py --audio ./test-2.mp3 --language en --benchmark

python main.py --audio ./test-2.mp3 --language en --temperature 0.6 --top-p 0.9 --top-k 10 --benchmark


 python3 handler.py --test_input '{"input": {"prompt": "Test prompt"}}'

python main_sta.py --audio ./test-2.mp3 --language en --temperature 0.6 --top-p 0.9 --top-k 10 --task translate 
### Flags
--language en (or yo, etc.)
--task translate



# Issues i encountered:
1. No Max issue with audio files. installing ffmpeg solved it.
```sh
brew install ffmpeg
```




```sh
# 1. Clone or upload your code to the pod
git clone <your-repo-url>
cd live-caption/server-2

# 2. Create venv and install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install websockets faster-whisper  # for websocket server

# 3. Start the WebSocket server
python websocket_server.py
```

## Docker Deployment (GPU)

```sh
cd server-2

# Build and run with docker-compose (recommended)
docker compose up -d --build

# Or build and run manually
docker build -t whisper-websocket .
docker run -d --gpus all -p 8765:8765 --name whisper-ws whisper-websocket

# View logs
docker logs -f whisper-websocket

# Stop
docker compose down
```

**Requirements:**
- NVIDIA GPU with CUDA 12.x
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)