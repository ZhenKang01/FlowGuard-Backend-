from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import uvicorn

# 1. Redefine the exact same model architecture so PyTorch knows how to load the weights
class FlowGuardAnomalyDetector(nn.Module):
    def __init__(self):
        super(FlowGuardAnomalyDetector, self).__init__()
        self.layer1 = nn.Linear(2, 16)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 8)
        self.output_layer = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output_layer(x))
        return x

# 2. Initialize the API and load the trained weights
app = FastAPI(title="FlowGuard AI API")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Note: Change to your Vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = FlowGuardAnomalyDetector()
model.load_state_dict(torch.load('flowguard_model.pth'))
model.eval() # Lock the model into evaluation mode (no more training)

# 3. Define the expected JSON payload from the frontend
class SensorData(BaseModel):
    hour: int
    flow_rate: float

# 4. Create the prediction endpoint
@app.post("/predict")
def predict_leak(data: SensorData):
    # Note: In production, apply the exact MinMaxScaler used in step 3.
    # For this prototype, we approximate the scaling (assuming max flow was ~150)
    scaled_flow = data.flow_rate / 150.0 
    
    # Convert input to PyTorch tensor
    input_tensor = torch.tensor([[float(data.hour), scaled_flow]], dtype=torch.float32)
    
    # Generate prediction
    with torch.no_grad():
        probability = model(input_tensor).item()
        
    return {
        "leak_probability": round(probability, 4),
        "is_leak_detected": probability > 0.5,
        "safety_protocol": "Acknowledge & Dispatch required." if probability > 0.5 else "Normal."
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)