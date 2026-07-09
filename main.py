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
    allow_origins=["*"], # Note: Change to your Vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Define the expected JSON payload from the frontend
class SensorData(BaseModel):
    hour: int
    flow_rate: float

# 4. Create the prediction endpoint
@app.post("/predict")
def predict_leak(data: SensorData):
    if not weights:
        return {"error": "Weights not found."}

    scaled_flow = data.flow_rate / 150.0 
    
    # Forward pass in pure Python
    x = [float(data.hour), scaled_flow]
    
    x = matmul(weights['layer1.weight'], x, weights['layer1.bias'])
    x = relu(x)
    
    x = matmul(weights['layer2.weight'], x, weights['layer2.bias'])
    x = relu(x)
    
    x = matmul(weights['output_layer.weight'], x, weights['output_layer.bias'])
    probability = sigmoid(x[0])
        
    return {
        "leak_probability": round(probability, 4),
        "is_leak_detected": probability > 0.5,
        "safety_protocol": "Acknowledge & Dispatch required." if probability > 0.5 else "Normal."
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)