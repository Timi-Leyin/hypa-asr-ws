import runpod

def handler(event):
    input_data = event["input"]
    prompt = input_data.get("prompt")
    return {"status": True, "message": f"Received prompt: {prompt}"}

runpod.serverless.start({"handler": handler})  