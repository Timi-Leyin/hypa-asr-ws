import { useCallback, useEffect, useRef, useState } from "react";

type WsStatus = "idle" | "connecting" | "open" | "closed" | "error";

type UseAudioWebSocketOptions = {
  url: string;
  autoConnect?: boolean;
  onEvent?: (event: unknown) => void;
  onOpen?: () => void;
  onClose?: (event: CloseEvent) => void;
  onError?: (event: Event) => void;
};

export function useAudioWebSocket({
  url,
  autoConnect = false,
  onEvent,
  onOpen,
  onClose,
  onError,
}: UseAudioWebSocketOptions) {
  const wsRef = useRef<WebSocket | null>(null);
  const [status, setStatus] = useState<WsStatus>("idle");
  const [lastEvent, setLastEvent] = useState<unknown>(null);

  const disconnect = useCallback(() => {
    const ws = wsRef.current;
    if (ws) {
      ws.close();
      wsRef.current = null;
    }
    setStatus("closed");
  }, []);

  const connect = useCallback(() => {
    if (!url) {
      setStatus("error");
      return;
    }

    const current = wsRef.current;
    if (current && (current.readyState === WebSocket.OPEN || current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    setStatus("connecting");
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("open");
      onOpen?.();
    };

    ws.onmessage = (messageEvent) => {
      let parsed: unknown = messageEvent.data;
      if (typeof messageEvent.data === "string") {
        try {
          parsed = JSON.parse(messageEvent.data);
        } catch {
          parsed = messageEvent.data;
        }
      }

      setLastEvent(parsed);
      onEvent?.(parsed);
    };

    ws.onclose = (closeEvent) => {
      setStatus("closed");
      onClose?.(closeEvent);
    };

    ws.onerror = (errorEvent) => {
      setStatus("error");
      onError?.(errorEvent);
    };
  }, [onClose, onError, onEvent, onOpen, url]);

  const sendJson = useCallback((payload: unknown) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return false;
    }

    ws.send(JSON.stringify(payload));
    return true;
  }, []);

  const sendAudioChunk = useCallback((chunkBase64: string, extra?: Record<string, unknown>) => {
    return sendJson({
      type: "audio_chunk",
      audio: chunkBase64,
      ...extra,
    });
  }, [sendJson]);

  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      const ws = wsRef.current;
      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        ws.close();
      }
      wsRef.current = null;
    };
  }, [autoConnect, connect]);

  return {
    status,
    isConnected: status === "open",
    lastEvent,
    connect,
    disconnect,
    sendJson,
    sendAudioChunk,
  };
}
