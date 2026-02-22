'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useVoiceStream } from 'voice-stream';
import type { ElevenLabsWebSocketEvent } from '../ws';

const sendMessage = (websocket: WebSocket, request: object) => {
    if (websocket.readyState !== WebSocket.OPEN) {
        return;
    }
    websocket.send(JSON.stringify(request));
};

export const useAgentConversation = () => {
    const websocketRef = useRef<WebSocket>(null);
    const [isConnected, setIsConnected] = useState<boolean>(false);

    const { startStreaming, stopStreaming } = useVoiceStream({
        onAudioChunked: (audioData) => {
            if (!websocketRef.current) return;
            //   console.log(audioData)
            sendMessage(websocketRef.current, {
                user_audio_chunk: audioData,
            });
        },
    });

    const startConversation = useCallback(async () => {
        if (isConnected) return;

        const websocket = new WebSocket("wss://api.elevenlabs.io/v1/convai/conversation?agent_id=agent_1701khttpeyje61rqc8z0hmw2drr");

        websocket.onopen = async () => {
            setIsConnected(true);
            sendMessage(websocket, {
                type: "conversation_initiation_client_data",
            });
            await startStreaming();
        };

        // websocket.send({
        //     type: "pong",
        //     event_id: Math.floor(Math.random() * 1000000),
        // })

        websocket.onmessage = async (event) => {
            const data = JSON.parse(event.data) as ElevenLabsWebSocketEvent;

            // Handle ping events to keep connection alive
            if (data.type === "ping") {
                setTimeout(() => {
                    sendMessage(websocket, {
                        type: "pong",
                        event_id: data.ping_event.event_id,
                    });
                }, data.ping_event.ping_ms);
            }

            if (data.type === "user_transcript") {
                const { user_transcription_event } = data;
                console.log("User transcript", user_transcription_event.user_transcript);
            }

            if (data.type === "agent_response") {
                const { agent_response_event } = data;
                console.log("Agent response", agent_response_event.agent_response);
            }

            if (data.type === "agent_response_correction") {
                const { agent_response_correction_event } = data;
                console.log("Agent response correction", agent_response_correction_event.corrected_agent_response);
            }

            if (data.type === "interruption") {
                // Handle interruption
            }

            if (data.type === "audio") {
                const { audio_event } = data;
                // Implement your own audio playback system here
                // Note: You'll need to handle audio queuing to prevent overlapping
                // as the WebSocket sends audio events in chunks
            }
        };

        websocketRef.current = websocket;

        websocket.onclose = async () => {
            websocketRef.current = null;
            setIsConnected(false);
            stopStreaming();
        };

        websocket.onerror = (error) => {
            console.error("WebSocket error", error);
            websocket.close();
        }
    }, [startStreaming, isConnected, stopStreaming]);

    const stopConversation = useCallback(async () => {
        if (!websocketRef.current) return;
        websocketRef.current.close();
    }, []);

    useEffect(() => {
        return () => {
            if (websocketRef.current) {
                websocketRef.current.close();
            }
        };
    }, []);

    return {
        startConversation,
        stopConversation,
        isConnected,
    };
};
