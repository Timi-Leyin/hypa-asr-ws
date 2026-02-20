import runpod

def handler(event):
    input_data = event["input"]
    
    return {"status": True}

runpod.serverless.start({"handler": handler})  