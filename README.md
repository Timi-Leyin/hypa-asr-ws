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