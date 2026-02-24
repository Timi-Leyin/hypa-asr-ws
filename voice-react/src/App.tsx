import { useCallback, useMemo, useState } from "react";
import { useVoiceStream } from "voice-stream";
import { encodeWAV, useWs } from "./hooks/use-ws";
import { ElevenLabsStreaming } from "./Elevenlabs";

function base64ToBytes(base64: string): Uint8Array {
  const normalized = base64.includes(",") ? base64.split(",", 2)[1] : base64;
  const binary = atob(normalized);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

function isWav(bytes: Uint8Array): boolean {
  return (
    bytes.length >= 12 &&
    bytes[0] === 0x52 && // R
    bytes[1] === 0x49 && // I
    bytes[2] === 0x46 && // F
    bytes[3] === 0x46 && // F
    bytes[8] === 0x57 && // W
    bytes[9] === 0x41 && // A
    bytes[10] === 0x56 && // V
    bytes[11] === 0x45 // E
  );
}

function bytesToFloat32LE(bytes: Uint8Array): Float32Array {
  const sampleCount = Math.floor(bytes.byteLength / 4);
  const samples = new Float32Array(sampleCount);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < sampleCount; index += 1) {
    samples[index] = view.getFloat32(index * 4, true);
  }
  return samples;
}

function bytesToFloat32FromInt16LE(bytes: Uint8Array): Float32Array {
  const sampleCount = Math.floor(bytes.byteLength / 2);
  const samples = new Float32Array(sampleCount);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < sampleCount; index += 1) {
    const value = view.getInt16(index * 2, true);
    samples[index] = value / 32768;
  }
  return samples;
}

function chunkBase64ToWavBytes(chunkBase64: string): Uint8Array {
  const bytes = base64ToBytes(chunkBase64);
  if (isWav(bytes)) {
    return bytes;
  }

  if (bytes.byteLength % 4 === 0) {
    return encodeWAV(bytesToFloat32LE(bytes));
  }

  if (bytes.byteLength % 2 === 0) {
    return encodeWAV(bytesToFloat32FromInt16LE(bytes));
  }

  return bytes;
}

function App() {
  const [receivedCount, setReceivedCount] = useState(0);
  const [sentCount, setSentCount] = useState(0);
  const [lastMessage, setLastMessage] = useState<string>("");

  const onWsMessage = useCallback((event: unknown) => {
    setReceivedCount((count) => count + 1);
    setLastMessage(typeof event === "string" ? event : JSON.stringify(event));
    console.log("Received WebSocket message", event);
  }, []);

  const { connect, disconnect, sendMessage, status } = useWs({
    url: "ws://213.173.102.133:19793",
    onMessage: onWsMessage,
  });

  const onAudioChunked = useCallback(
    (chunkBase64: string) => {
      try {
        const wavBytes = chunkBase64ToWavBytes(chunkBase64);
        const sent = sendMessage(wavBytes);
        if (sent) {
          setSentCount((count) => count + 1);
        }
      } catch (error) {
        console.error("Chunk conversion/send failed", error);
      }
    },
    [sendMessage],
  );

  const voiceStreamOptions = useMemo(
    () => ({
      onStartStreaming: () => {
        console.log("Streaming started");
      },
      onStopStreaming: () => {
        console.log("Streaming stopped");
      },
      onAudioChunked,
    }),
    [onAudioChunked],
  );

  const { startStreaming, stopStreaming, isStreaming } = useVoiceStream({
    ...voiceStreamOptions,
  });

  const startAll = useCallback(() => {
    connect();
    startStreaming();
  }, [connect, startStreaming]);

  const stopAll = useCallback(() => {
    stopStreaming();
    disconnect();
  }, [disconnect, stopStreaming]);
  // return <ElevenLabsStreaming />;
  return (
    <div>
      <div>WebSocket: {status}</div>
      <div>
        Sent chunks: {sentCount} | Received messages: {receivedCount}
      </div>
      <div>Last message: {JSON.stringify(lastMessage) || "-"}</div>
      <button onClick={connect}>Connect WS</button>

      <button
        onClick={() => {
          sendMessage({
            type: "ping",
            event_id: Math.floor(Math.random() * 1000000),
          });
        }}
      >
        Ping
      </button>

      <button
        onClick={() => {
          sendMessage("shutdown");
        }}
      >
        SHUTDOWN
      </button>

      <button onClick={disconnect}>Disconnect WS</button>
      <button onClick={startAll} disabled={isStreaming}>
        Start Recording
      </button>
      <button onClick={stopAll} disabled={!isStreaming}>
        Stop Recording
      </button>
      {/* <pre>{JSON.stringify(lastEvent, null, 2)}</pre> */}
    </div>
  );
}

export default App;
