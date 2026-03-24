import torch
import torch.nn as nn
import joblib
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# Initialize API
app = FastAPI(
    title="Housing Price Prediction API",
    description="Predict California housing prices using a trained PyTorch model",
    version="1.0"
)

# Load scaler
scaler = joblib.load("scaler.pkl")

# Load templates (UI)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Input schema (for API)
class HousingInput(BaseModel):
    MedInc: float = Field(..., example=8.3)
    HouseAge: float = Field(..., example=41)
    AveRooms: float = Field(..., example=6.5)
    AveBedrms: float = Field(..., example=1.0)
    Population: float = Field(..., example=322)
    AveOccup: float = Field(..., example=2.5)
    Latitude: float = Field(..., example=37.88)
    Longitude: float = Field(..., example=-122.23)

# Define model (MUST match training)
model = nn.Sequential(
    nn.Linear(8, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# Load trained weights
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# ---------------------------
# 🌐 UI ROUTE (Homepage)
# ---------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------------------------
# 📊 API ROUTE (Prediction)
# ---------------------------
@app.post("/predict")
def predict(data: HousingInput):

    # Convert structured input → list
    features = [
        data.MedInc,
        data.HouseAge,
        data.AveRooms,
        data.AveBedrms,
        data.Population,
        data.AveOccup,
        data.Latitude,
        data.Longitude
    ]

    # Scale input (CRITICAL)
    scaled = scaler.transform([features])

    # Convert to tensor
    x = torch.tensor(scaled, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        prediction = model(x).item()

    return {
        "predicted_house_value": round(prediction, 2),
        "estimated_price_usd": int(prediction * 100000)
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request
    })