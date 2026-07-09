from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import json
import math
import os

# 1. Pure Python inference engine (No PyTorch needed)
def relu(x):
    return [max(0, val) for val in x]

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def matmul(W, x, b):
    out = []
    for i in range(len(W)):
        val = sum(W[i][j] * x[j] for j in range(len(x))) + b[i]
        out.append(val)
    return out

# Load weights from the JSON file
weights = {}
weights_path = os.path.join(os.path.dirname(__file__), 'model_weights.json')
if os.path.exists(weights_path):
    with open(weights_path, 'r') as f:
        weights = json.load(f)

# 2. Initialize the API
app = FastAPI(title="FlowGuard AI API")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Create the prediction endpoint
from fastapi import Request

@app.post("/predict")
async def predict_leak(request: Request):
    form = await request.form()
    
    hour_str = form.get("hour")
    flow_rate_str = form.get("flow_rate")
    
    if not hour_str or not flow_rate_str:
        return {"error": "Missing hour or flow_rate"}
        
    hour = int(hour_str)
    flow_rate = float(flow_rate_str)

    if not weights:
        return {"error": "Weights not found."}

    scaled_flow = flow_rate / 150.0 
    
    # Forward pass in pure Python
    x = [float(hour), scaled_flow]
    
    x = matmul(weights['layer1.weight'], x, weights['layer1.bias'])
    x = relu(x)
    
    x = matmul(weights['layer2.weight'], x, weights['layer2.bias'])
    x = relu(x)
    
    x = matmul(weights['output_layer.weight'], x, weights['output_layer.bias'])
    probability = sigmoid(x[0])
    
    # 2. Roboflow Image Inference (Optional)
    import os
    import httpx
    
    image = form.get("image")
    roboflow_leak = False
    roboflow_confidence = None
    roboflow_message = "No image provided."

    if image and hasattr(image, 'filename') and image.filename:
        rf_api_key = os.getenv("ROBOFLOW_API_KEY")
        rf_endpoint = os.getenv("ROBOFLOW_MODEL_ENDPOINT") # e.g. "my-project/1"

        if not rf_api_key or not rf_endpoint:
            roboflow_message = "Roboflow API Key or Endpoint not configured in Vercel environment variables."
        else:
            try:
                image_bytes = await image.read()
                url = f"https://detect.roboflow.com/{rf_endpoint}?api_key={rf_api_key}"
                resp = httpx.post(
                    url,
                    files={"file": (image.filename or "upload.jpg", image_bytes, "application/octet-stream")},
                    timeout=30.0,
                    verify=False,
                )
                
                if resp.status_code == 200:
                    rf_data = resp.json()
                    predictions = rf_data.get("predictions", [])
                    # Look for "leak" class
                    leak_preds = [p for p in predictions if p.get("class", "").lower() == "leak"]
                    if leak_preds:
                        roboflow_leak = True
                        roboflow_confidence = max(p.get("confidence", 0) for p in leak_preds)
                        roboflow_message = f"Leak detected in image (Confidence: {roboflow_confidence:.2f})"
                    else:
                        roboflow_message = "No leak detected in image."
                else:
                    roboflow_message = f"Roboflow API error: {resp.status_code} - {resp.text}"
            except Exception as e:
                roboflow_message = f"Failed to call Roboflow: {str(e)}"

    # 3. Combine Results
    is_leak_detected = probability > 0.5 or roboflow_leak
        
    return {
        "leak_probability": round(probability, 4),
        "is_leak_detected": is_leak_detected,
        "pytorch_detected": probability > 0.5,
        "roboflow_detected": roboflow_leak,
        "roboflow_message": roboflow_message,
        "safety_protocol": "Acknowledge & Dispatch required." if is_leak_detected else "Normal."
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)