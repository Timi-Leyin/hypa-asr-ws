#!/usr/bin/env python3
"""
Shutdown lifecycle:
  - SIGTERM / SIGINT → sets _stop_event → run_in_executor unblocks →
    websockets closes all connections → thread exits → handler() returns cleanly.
"""
import asyncio
import os
import signal
import threading

import runpod
import websockets


from transcriber import TranscriptionServer, MODEL_PATH, log

WS_HOST = "0.0.0.0"
WS_PORT = 8765

_stop_event: threading.Event = threading.Event()


def _handle_signal(signum, frame) -> None:
    log.info("Signal %s received – initiating shutdown …", signal.Signals(signum).name)
    _stop_event.set()


async def _run_ws_server(server: TranscriptionServer) -> None:
    loop = asyncio.get_running_loop()
    async with websockets.serve(
        server.handle_client,
        WS_HOST,
        WS_PORT,
        max_size=10 * 1024 * 1024,
        ping_interval=20,
        ping_timeout=10,
    ):
        log.info("WebSocket server listening on ws://%s:%d", WS_HOST, WS_PORT)
        await loop.run_in_executor(None, _stop_event.wait)

    log.info("WebSocket server stopped.")


def _start_server_thread(server: TranscriptionServer) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run_ws_server(server))
    finally:
        loop.close()



def handler(event):
    _stop_event.clear()
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    public_ip   = os.environ.get("RUNPOD_PUBLIC_IP", "localhost")
    public_port = int(os.environ.get("RUNPOD_TCP_PORT_8765", "8765"))
    public_http_port = int(os.environ.get("RUNPOD_HTTP_PORT_8766", "8766"))
    print(f"Handler started | Public IP: {public_ip}, TCP Port: {public_port}, HTTP Port: {public_http_port}")
    runpod.serverless.progress_update(
        event,
        f"Loading model: {MODEL_PATH} | Public IP: {public_ip}, TCP Port: {public_port}",
    )


    server = TranscriptionServer(shutdown_event=_stop_event)

    runpod.serverless.progress_update(event, f"Model ready | Public IP: {public_ip}, TCP Port: {public_port}")
    log.info("Public endpoint: %s:%d", public_ip, public_port)

    ws_thread = threading.Thread(
        target=_start_server_thread, args=(server,), daemon=True, name="ws-server",
    )
    ws_thread.start()
    ws_thread.join()

    log.info("Handler returning cleanly.")
    return {
        "message":     "Transcription server stopped",
        "public_ip":   public_ip,
        "public_port": public_port,
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
