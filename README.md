python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt

python main.py --audio /absolute/path/to/audio.wav

python main_faster.py --audio ./test-2.mp3 --language en --benchmark

python main.py --audio ./test-2.mp3 --language en --temperature 0.6 --top-p 0.9 --top-k 10 --benchmark


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