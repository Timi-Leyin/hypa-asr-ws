#!/usr/bin/env python3
"""
WebSocket Client Test
Simulates sending audio chunks to the transcription server
"""
import asyncio
import json
import wave
import websockets
from pathlib import Path
import sys


async def test_audio_file(audio_path, server_url="ws://localhost:8765"):
    """Test by sending an audio file in chunks"""
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        return
    
    print(f"🎤 Testing with: {audio_path}")
    print(f"📡 Connecting to: {server_url}\n")
    
    async with websockets.connect(server_url) as websocket:
        # Wait for connection message
        response = await websocket.recv()
        data = json.loads(response)
        print(f"✓ {data.get('message')}")
        print(f"  Model: {data.get('model')}\n")
        
        # Optional: Set language
        await websocket.send(json.dumps({
            "type": "config",
            "language": "en"
        }))
        config_response = await websocket.recv()
        print(f"✓ Config: {json.loads(config_response)}\n")
        
        # Read audio file
        print("📤 Sending audio chunks...\n")
        with wave.open(str(audio_path), 'rb') as wf:
            chunk_size = wf.getframerate() * 1  # 1 second chunks
            
            while True:
                frames = wf.readframes(chunk_size)
                if not frames:
                    break
                
                # Send chunk
                await websocket.send(frames)
                print(".", end="", flush=True)
                
                # Check for responses
                try:
                    response = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=0.1
                    )
                    data = json.loads(response)
                    
                    if data["type"] == "transcription":
                        print(f"\n\n✓ Transcription:")
                        print(f"  Text: {data['text']}")
                        print(f"  Duration: {data['duration']:.2f}s")
                        print(f"  Processing: {data['processing_time']:.2f}s")
                        print(f"  RTF: {data['rtf']:.2f}x")
                        print()
                    elif data["type"] == "buffering":
                        pass  # Buffering status
                    elif data["type"] == "error":
                        print(f"\n❌ Error: {data['message']}\n")
                
                except asyncio.TimeoutError:
                    pass  # No response yet
        
        print(f"\n\n✓ Finished sending audio")
        
        # Wait a bit for final transcriptions
        print("⏳ Waiting for final transcriptions...\n")
        try:
            while True:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)
                
                if data["type"] == "transcription":
                    print(f"✓ Final transcription:")
                    print(f"  Text: {data['text']}")
                    print(f"  RTF: {data['rtf']:.2f}x\n")
        except asyncio.TimeoutError:
            print("✓ No more transcriptions\n")


async def test_streaming_simulation(server_url="ws://localhost:8765"):
    """Simulate live microphone streaming"""
    print("🎤 Simulating live microphone stream")
    print(f"📡 Connecting to: {server_url}\n")
    
    async with websockets.connect(server_url) as websocket:
        # Wait for connection
        response = await websocket.recv()
        print(f"✓ {json.loads(response).get('message')}\n")
        
        print("🔴 Simulating 10 seconds of audio streaming...")
        print("   (In real app, this would be live microphone data)\n")
        
        # Simulate sending chunks
        import numpy as np
        sample_rate = 16000
        chunk_duration = 0.5  # 500ms chunks
        
        for i in range(20):  # 20 chunks * 0.5s = 10 seconds
            # Generate silence (in real app, this is microphone data)
            chunk = np.zeros(int(sample_rate * chunk_duration), dtype=np.float32)
            
            # Send chunk
            await websocket.send(chunk.tobytes())
            await asyncio.sleep(chunk_duration)  # Simulate real-time
            
            # Check for responses
            try:
                response = await asyncio.wait_for(
                    websocket.recv(),
                    timeout=0.01
                )
                data = json.loads(response)
                
                if data["type"] == "transcription":
                    print(f"[{i*chunk_duration:.1f}s] Transcription: {data['text']}")
            except asyncio.TimeoutError:
                pass
        
        print("\n✓ Simulation complete\n")


async def test_ping(server_url="ws://localhost:8765"):
    """Test server connectivity"""
    print(f"🔍 Pinging server: {server_url}\n")
    
    try:
        async with websockets.connect(server_url) as websocket:
            # Wait for connection
            response = await websocket.recv()
            print(f"✓ {json.loads(response).get('message')}")
            
            # Send ping
            await websocket.send(json.dumps({"type": "ping"}))
            
            # Wait for pong
            response = await websocket.recv()
            data = json.loads(response)
            
            if data["type"] == "pong":
                print(f"✓ Server is alive")
                print(f"  Timestamp: {data['timestamp']}\n")
    except Exception as e:
        print(f"❌ Connection failed: {e}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_websocket.py <audio_file>     # Test with audio file")
        print("  python test_websocket.py --simulate        # Simulate streaming")
        print("  python test_websocket.py --ping            # Test connection")
        print("\nExample:")
        print("  python test_websocket.py ./test.wav")
        sys.exit(1)
    
    command = sys.argv[1]
    
    try:
        if command == "--simulate":
            asyncio.run(test_streaming_simulation())
        elif command == "--ping":
            asyncio.run(test_ping())
        else:
            # Assume it's an audio file path
            asyncio.run(test_audio_file(command))
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
