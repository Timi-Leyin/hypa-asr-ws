import { useCallback, useRef, useState } from "react";

interface UseWsOptions {
    url: string;
    onMessage?: (event: any) => void;
}



const SAMPLE_RATE = 16000;


export function encodeWAV(samples: Float32Array) {
    const numSamples = samples.length;
    const buffer = new ArrayBuffer(44 + numSamples * 2);
    const view = new DataView(buffer);

    const writeStr = (off: number, str: string) => { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); };
    const writeU32 = (off: number, v: number) => view.setUint32(off, v, true);
    const writeU16 = (off: number, v: number) => view.setUint16(off, v, true);

    writeStr(0, 'RIFF');
    writeU32(4, 36 + numSamples * 2);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    writeU32(16, 16);
    writeU16(20, 1);               // PCM
    writeU16(22, 1);               // mono
    writeU32(24, SAMPLE_RATE);
    writeU32(28, SAMPLE_RATE * 2);
    writeU16(32, 2);               // block align
    writeU16(34, 16);              // bits/sample
    writeStr(36, 'data');
    writeU32(40, numSamples * 2);

    let off = 44;
    for (let i = 0; i < numSamples; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        off += 2;
    }
    return new Uint8Array(buffer);
}



export const useWs = ({ url, onMessage }: UseWsOptions) => {
    const wsRef = useRef<WebSocket | null>(null);
    const [status, setStatus] = useState("idle");

    const connect = useCallback(() => {
        if (!url) {
            console.error("WebSocket URL is required");
            return;
        }

        const current = wsRef.current;
        if (current && (current.readyState === WebSocket.OPEN || current.readyState === WebSocket.CONNECTING)) {
            console.warn("WebSocket is already connected or connecting");
            return;
        }

        const ws = new WebSocket(url);
        wsRef.current = ws;
        setStatus("connecting");

        ws.onopen = () => {
            console.log("WebSocket connected");
            setStatus("open");
        };

        ws.onmessage = (messageEvent) => {
            let event = messageEvent.data;
            if (typeof messageEvent.data === "string") {
                try {
                    event = JSON.parse(messageEvent.data);
                } catch {
                    event = messageEvent.data;
                }
            }

            if (onMessage) {
                onMessage(event);
            }
            // if (msg.type === 'caption' || msg.type === 'silence' || msg.type === 'busy') {}
        }
        ws.onclose = (closeEvent) => {
            console.log("WebSocket closed", closeEvent);
            setStatus("closed");
        };

        ws.onerror = (errorEvent) => {
            console.error("WebSocket error", errorEvent);
            setStatus("error");
        };
    }, [url, onMessage]);

    const sendMessage = useCallback((message: any) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            console.error("WebSocket is not connected");
            return false;
        }
        wsRef.current?.send(message);
        return true;
    }, []);

    const disconnect = useCallback(() => {
        const ws = wsRef.current;
        if (ws) {
            ws.close();
            wsRef.current = null;
        }
        setStatus("closed");
    }, []);

    return {
        status,
        connect,
        sendMessage,
        disconnect,
    };
};
