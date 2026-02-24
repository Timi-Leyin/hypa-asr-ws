#!/usr/bin/env python3
"""
Shutdown lifecycle:
  - SIGTERM / SIGINT → sets _stop_event → asyncio loop cancels the server
    coroutine → websockets closes all connections → thread exits → handler()
    returns cleanly.
"""
import asyncio
import os
import signal
import threading

import runpod
import websockets

from transcriber import (
    TranscriptionServer,
    MODEL_PATH,
    log,
)

_stop_event: threading.Event = threading.Event()


def _handle_signal(signum, frame) -> None:
    log.info("Signal %s received – initiating shutdown …", signal.Signals(signum).name)
    _stop_event.set()


WS_HOST = "0.0.0.0"
WS_PORT = 8765


async def _run_ws_server(server: TranscriptionServer, public_ip: str, public_port: int) -> None:
    """Start the WebSocket server and run until _stop_event is set."""
    stop_future: asyncio.Future = asyncio.get_running_loop().create_future()

    # Poll the threading.Event from inside the asyncio loop.
    async def _watch_stop() -> None:
        while not _stop_event.is_set():
            await asyncio.sleep(0.5)
        if not stop_future.done():
            stop_future.set_result(None)

    async with websockets.serve(
        server.handle_client,
        WS_HOST,
        WS_PORT,
        max_size=10 * 1024 * 1024,
        ping_interval=20,
        ping_timeout=10,
    ):
        log.info(
            "WebSocket server listening on ws://%s:%d (public: %s:%d)",
            WS_HOST, WS_PORT, public_ip, public_port,
        )
        watcher = asyncio.ensure_future(_watch_stop())
        try:
            await stop_future          # blocks until shutdown requested
        finally:
            watcher.cancel()

    log.info("WebSocket server stopped.")


def _start_server_thread(server: TranscriptionServer, public_ip: str, public_port: int) -> None:
    """Run the asyncio event loop in a dedicated thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run_ws_server(server, public_ip, public_port))
    finally:
        loop.close()


def handler(event):
    global _stop_event
    _stop_event.clear() 
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)

    public_ip   = os.environ.get("RUNPOD_PUBLIC_IP", "localhost")
    public_port = int(os.environ.get("RUNPOD_TCP_PORT_8765", "8765"))

    runpod.serverless.progress_update(
        event,
        f"Loading model: {MODEL_PATH} | Public IP: {public_ip}, TCP Port: {public_port}",
    )

    server = TranscriptionServer(shutdown_event=_stop_event)

    runpod.serverless.progress_update(
        event,
        f"Model ready. WebSocket server starting on {WS_HOST}:{WS_PORT} (public: {public_ip}:{public_port})",
    )

    ws_thread = threading.Thread(
        target=_start_server_thread, args=(server, public_ip, public_port), daemon=True, name="ws-server",
    )
    ws_thread.start()
    ws_thread.join()  # blocks until _stop_event fires and the loop exits

    log.info("Handler returning cleanly.")
    return {
        "message":     "Transcription server stopped",
        "public_ip":   public_ip,
        "public_port": public_port,
    }



if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
