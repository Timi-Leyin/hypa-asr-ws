from websocket_server import WebsocketServer
import runpod
import os

shutdown_flag = False

def on_message(client, server, message):
    global shutdown_flag
    print(f"Received: {message}")
    server.send_message(client, f"Echo: {message}")

    if message.strip().lower() == "shutdown":
        print("Shutdown command received. Stopping WebSocket server...")
        shutdown_flag = True
        server.shutdown()

def start_websocket():
    global shutdown_flag
    server = WebsocketServer(host="0.0.0.0", port=8765)
    server.set_fn_message_received(on_message)
    print("WebSocket server started on port 8765...")

    while not shutdown_flag:
        server.run_forever()

    return "WebSocket server stopped successfully"

def handler(event):
    public_ip = os.environ.get('RUNPOD_PUBLIC_IP', 'localhost')
    tcp_port = int(os.environ.get('RUNPOD_TCP_PORT_8765', '8765'))
    tcp_port_alt = int(os.environ.get('RUNPOD_TCP_PORT', '8765'))

    runpod.serverless.progress_update(event, f"Public IP: {public_ip}, TCP Port: {tcp_port}, TCP Port Alt: {tcp_port_alt}")

    result = start_websocket()

    return {
        "message": result,
        "public_ip": public_ip,
        "tcp_port_alt": tcp_port_alt,
        "tcp_port": tcp_port
    }

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})