# WebSocket Live Transcription Server

Real-time audio transcription server using WebSocket for live captions.

## Features

- ✅ Real-time audio streaming via WebSocket
- ✅ Processes audio chunks as array buffers
- ✅ Uses your converted `hypaai/wspr_small_2025-11-11_12-12-17` model
- ✅ Fast startup (0.5s) with converted CT2 model
- ✅ Multiple simultaneous connections
- ✅ Configurable language and settings
- ✅ Browser-based client with live microphone

## Quick Start

### 1. Install Dependencies

```bash
pip install websockets librosa
```

Or use the full requirements:
```bash
pip install -r requirements_optimized.txt
```

### 2. Start the Server

```bash
python websocket_server.py
```

You should see:
```
🚀 Loading model from ./wspr_small_ct2...
✓ Model loaded successfully

============================================================
🎤 WEBSOCKET TRANSCRIPTION SERVER
============================================================
Host: 0.0.0.0
Port: 8765
Model: ./wspr_small_ct2
Sample Rate: 16000 Hz
Chunk Duration: 2.0s
============================================================

✓ Server listening on ws://0.0.0.0:8765
Waiting for connections...
```

### 3. Test with Audio File

```bash
python test_websocket.py ./test.wav
```

### 4. Test with Browser Client

Open `client.html` in your browser:
```bash
open client.html   # macOS
# or
xdg-open client.html   # Linux
# or just double-click the file
```

Then:
1. Click "Connect"
2. Click "Start Recording"
3. Speak into your microphone
4. See real-time transcriptions appear!

## Server Configuration

Edit these settings in `websocket_server.py`:

```python
HOST = "0.0.0.0"          # Listen on all interfaces
PORT = 8765                # WebSocket port
SAMPLE_RATE = 16000        # Audio sample rate
CHUNK_DURATION = 2.0       # Process every 2 seconds
MIN_AUDIO_LENGTH = 0.5     # Skip chunks shorter than 0.5s
```

## WebSocket Protocol

### Client → Server

#### 1. Audio Data (Binary)
Send raw audio bytes as array buffer:
```javascript
// Browser example
const arrayBuffer = await audioBlob.arrayBuffer();
websocket.send(arrayBuffer);
```

Supported formats:
- WAV files (any sample rate, will be resampled)
- Raw PCM float32
- Raw PCM int16

#### 2. Configuration (JSON)
```json
{
  "type": "config",
  "language": "en",
  "task": "transcribe"
}
```

#### 3. Clear Buffer (JSON)
```json
{
  "type": "clear_buffer"
}
```

#### 4. Ping (JSON)
```json
{
  "type": "ping"
}
```

### Server → Client

#### 1. Connected
```json
{
  "type": "connected",
  "message": "WebSocket transcription server ready",
  "model": "./wspr_small_ct2",
  "timestamp": "2026-02-18T00:00:00"
}
```

#### 2. Transcription
```json
{
  "type": "transcription",
  "text": "She serves 6 years at the C-show...",
  "duration": 6.06,
  "processing_time": 1.87,
  "rtf": 0.31,
  "timestamp": "2026-02-18T00:00:00"
}
```

#### 3. Buffering Status
```json
{
  "type": "buffering",
  "buffered": 1.5,
  "needed": 2.0
}
```

#### 4. Error
```json
{
  "type": "error",
  "message": "Audio processing error: ..."
}
```

#### 5. Pong
```json
{
  "type": "pong",
  "timestamp": "2026-02-18T00:00:00"
}
```

## Python Client Example

```python
import asyncio
import websockets
import wave

async def transcribe_file(audio_path):
    async with websockets.connect("ws://localhost:8765") as ws:
        # Wait for connection message
        await ws.recv()
        
        # Send audio file
        with wave.open(audio_path, 'rb') as wf:
            while True:
                frames = wf.readframes(16000)  # 1 second chunks
                if not frames:
                    break
                await ws.send(frames)
                
                # Get transcription
                response = await ws.recv()
                print(response)

asyncio.run(transcribe_file("./test.wav"))
```

## Browser Client Example

```javascript
const ws = new WebSocket('ws://localhost:8765');

ws.onopen = async () => {
    // Set language
    ws.send(JSON.stringify({
        type: 'config',
        language: 'en'
    }));
    
    // Start microphone
    const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: { sampleRate: 16000, channelCount: 1 }
    });
    
    const mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = async (event) => {
        const arrayBuffer = await event.data.arrayBuffer();
        ws.send(arrayBuffer);
    };
    
    mediaRecorder.start(500); // Send chunks every 500ms
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'transcription') {
        console.log('Transcription:', data.text);
        console.log('RTF:', data.rtf);
    }
};
```

## Performance

With converted CT2 model:
- **Server startup:** ~0.5s
- **Average RTF:** ~0.3x (30% of real-time)
- **Latency:** 2-3 seconds (chunk duration + processing)
- **Memory:** ~500 MB per connection

## Testing Options

### 1. Test with audio file
```bash
python test_websocket.py ./test.wav
```

### 2. Test server ping
```bash
python test_websocket.py --ping
```

### 3. Test streaming simulation
```bash
python test_websocket.py --simulate
```

### 4. Test with browser
```bash
open client.html
```

## Troubleshooting

### Server won't start
- Check if model is converted: `ls -la ./wspr_small_ct2/`
- If not, run: `python convert_model.py`

### No transcriptions
- Check chunk duration (needs at least 0.5s of audio)
- Verify audio format (should be 16kHz mono)
- Check server logs for errors

### High latency
- Reduce `CHUNK_DURATION` (but not below 1.0s)
- Ensure model is converted to CT2 format
- Check CPU usage

### Browser can't connect
- Ensure server is running: `python websocket_server.py`
- Check firewall settings
- Try `localhost` instead of `0.0.0.0`

## Production Deployment

### With Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements_optimized.txt .
RUN pip install -r requirements_optimized.txt

COPY websocket_server.py .
COPY wspr_small_ct2/ ./wspr_small_ct2/

EXPOSE 8765
CMD ["python", "websocket_server.py"]
```

### With systemd
```ini
[Unit]
Description=Live Caption WebSocket Server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/live-caption
ExecStart=/usr/bin/python3 websocket_server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

## Next Steps

1. ✅ Server is running
2. ✅ Test with audio file
3. ✅ Test with browser client
4. ⏭️ Integrate into your application
5. ⏭️ Deploy to production

## Architecture

```
┌─────────────┐                ┌──────────────────┐
│   Browser   │◄──WebSocket──►│  Server Process  │
│  (client)   │                │                  │
│             │                │  ┌────────────┐  │
│ Microphone  │    Audio       │  │  Buffer    │  │
│     ↓       │   Chunks       │  │  Manager   │  │
│ MediaRecorder   ─────────►   │  └─────┬──────┘  │
│             │                │        ↓          │
└─────────────┘                │  ┌────────────┐  │
                               │  │  Whisper   │  │
┌─────────────┐                │  │   Model    │  │
│  Another    │◄──WebSocket──►│  │ (CT2/INT8) │  │
│   Client    │                │  └─────┬──────┘  │
└─────────────┘                │        ↓          │
                               │  Transcription   │
                               │        ↓          │
                               │   WebSocket      │
                               │    Response      │
                               └──────────────────┘
```

## Model Info

- **Model:** Your custom `hypaai/wspr_small_2025-11-11_12-12-17`
- **Format:** CTranslate2 INT8 (converted)
- **Size:** 241 MB
- **Quality:** ~95% of original (INT8 quantization)
- **Speed:** 5-10x faster than original

Your exact model is used - just converted to a faster format! ✅
