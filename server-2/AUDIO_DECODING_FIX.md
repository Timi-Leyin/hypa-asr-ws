# Audio Decoding Fix: WebM → Raw PCM

## Problem
The server was failing to decode audio blobs from the browser with:
```
Librosa failed: Error opening <_io.BytesIO object>: Format not recognised
WAV decode failed: file does not start with RIFF id
```

**Root Cause**: MediaRecorder produces fragmented WebM/Opus format when chunks are concatenated. WebM is a complex container format with:
- Initialization segments (must come first)
- Cluster timestamps
- Complex framing requirements

Simply concatenating MediaRecorder chunks with `new Blob(recordedChunks, {type: mimeType})` creates an invalid WebM file that none of the decoders can read.

## Solution: Switch to Raw PCM

Instead of encoding audio on the browser and decoding on the server, we now send **raw PCM (Pulse-Code Modulation) data** directly.

### How it works:

#### Client (Web Audio API)
```javascript
// Create Web Audio API pipeline
const audioContext = new AudioContext({ sampleRate: 16000 });
const source = audioContext.createMediaStreamSource(stream);
const processor = audioContext.createScriptProcessor(4096, 1, 1);

// Accumulate samples and send every 100ms
processor.onaudioprocess = (event) => {
    audioBuffer.push(new Float32Array(event.inputBuffer.getChannelData(0)));
};

// Convert Float32 to Int16 PCM and send
const combined = /* concatenate Float32 samples */;
const int16Array = new Int16Array(combined.length);
for (let i = 0; i < combined.length; i++) {
    int16Array[i] = Math.max(-32768, Math.min(32767, combined[i] * 0x7FFF));
}
ws.send(int16Array.buffer);  // Send as binary
```

**Advantages**:
- ✅ No encoding overhead → real-time capable
- ✅ Trivial to decode (just reinterpret bytes)
- ✅ Fixed sample rate (16kHz) and format (Int16 mono)
- ✅ Works with any browser (no codec negotiation needed)

#### Server (PCM Decoding)
```python
# Method 2 in bytes_to_audio_array() - tried early
audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
```

Decoding is a single line - just convert byte buffer to Int16, then normalize to float32.

## Format Details

| Property | Value |
|----------|-------|
| Sample Rate | 16 kHz |
| Bit Depth | 16-bit (Int16) |
| Channels | 1 (Mono) |
| Encoding | Raw PCM (no container) |
| Byte Order | Native (system dependent) |

## Decoder Fallback Order (websocket_server.py)

1. **WAV files** - Standard .wav with RIFF header
2. **Raw PCM (Int16)** - Direct from Web Audio API ← **NEW PRIMARY**
3. **Temp file → librosa** - For WebM/MP3 via ffmpeg
4. **BytesIO → librosa** - As last resort
5. **Raw float32** - Direct float reinterpretation
6. **Raise ValueError** - All methods failed

## Migration Notes

- **No model changes** - Whisper still transcribes the same way
- **No transcription changes** - Quality identical to before
- **No server setup changes** - Already had ffmpeg installed
- **Client-only breaking change** - Old WebM-based client won't work with new server

## Testing Checklist

- [ ] Start server: `python websocket_server.py`
- [ ] Open client: `client.html` in browser
- [ ] Click "Start Recording"
- [ ] Speak clearly (test phrase: "Hello, this is a test")
- [ ] Verify logs show:
  - `✓ Audio context: sr=16000Hz`
  - `✓ Raw int16 PCM: len=XXXX, RMS=X.XXXXXX`
  - `✓ Transcribed: final text`
- [ ] Check transcript appears in "Transcription Results" box
- [ ] Stop recording and repeat

## Performance Impact

- **Audio latency**: Reduced from 2s to 0.5s buffer window
- **CPU usage**: Very low (no encoding, direct memory buffer)
- **Memory**: Minimal (PCM is uncompressed but 16-bit mono at 16kHz is only ~32 KB/sec)
- **Network bandwidth**: ~32 KB/sec (larger than WebM/Opus but manageable for local network)

## Architecture Diagram

```
Browser                          Server
========                         ======
🎤 Microphone
  ↓
AudioContext (16kHz)
  ↓
ScriptProcessor (4096 samples)
  ↓
Accumulate Float32 samples
  ↓
Every 100ms: Convert to Int16 PCM
  ↓
ws.send(ArrayBuffer)  ────────→ [WS receive]
                                  ↓
                            bytes_to_audio_array()
                                  ↓
                            np.frombuffer(dtype=int16)
                                  ↓
                            Normalize to float32 [-1, 1]
                                  ↓
                            Accumulate in audio_buffer
                                  ↓
                            When buffer ≥ 0.5s:
                            concatenate + transcribe
                                  ↓
                            🎯 Speech to Text
                                  ↓
                                ws.send(JSON) ────────→ Display on page
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Raw int16 PCM failed: struct.error" | Byte count not multiple of 2 | Check client sends full Int16 samples |
| No audio chunking logged | Audio processor not running | Check browser DevTools console for errors |
| RMS = nan in logs | Silent audio | Speak louder or check microphone |
| "Transcription complete. Text: None" | Speech not detected | Increase buffer window or check VAD settings |

## References

- [Web Audio API - ScriptProcessor](https://developer.mozilla.org/en-US/docs/Web/API/ScriptProcessorNode)
- [PCM Wikipedia](https://en.wikipedia.org/wiki/Pulse-code_modulation)
- [NumPy ByteBuffer to Array](https://numpy.org/doc/stable/reference/generated/numpy.frombuffer.html)
