# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import numpy as np
from nbeats_pytorch.model import NBeatsNet

# ----------------------------
# Safe load for custom objects
# ----------------------------
from torch.serialization import add_safe_globals
from sklearn.preprocessing import StandardScaler

# Allow StandardScaler from sklearn
add_safe_globals([StandardScaler])

# ----------------------------
# App & device
# ----------------------------
app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Load model checkpoint
# ----------------------------
checkpoint = torch.load(
    "model/nbeats_pm25.pt",  # path to your .pt file
    map_location=device,
    weights_only=False        # required because checkpoint includes scaler
)

LOOKBACK = checkpoint["lookback"]
HORIZON = checkpoint["horizon"]

model = NBeatsNet(
    device=device,
    stack_types=(NBeatsNet.GENERIC_BLOCK,) * 3,
    nb_blocks_per_stack=3,
    backcast_length=LOOKBACK,
    forecast_length=HORIZON,
    hidden_layer_units=512,
    thetas_dim=(4, 4, 4)
).to(device)

model.load_state_dict(checkpoint["model_state"])
model.eval()

scaler = checkpoint["scaler"]

# ----------------------------
# Request model
# ----------------------------
class PM25Request(BaseModel):
    pm25_history: List[float]

# ----------------------------
# AQI conversion function
# ----------------------------
def pm25_to_aqi(pm):
    pm = float(pm)
    if pm <= 30: return (pm / 30) * 50
    elif pm <= 60: return 50 + (pm - 30) * 50 / 30
    elif pm <= 90: return 100 + (pm - 60) * 100 / 30
    elif pm <= 120: return 200 + (pm - 90) * 100 / 30
    elif pm <= 250: return 300 + (pm - 120) * 100 / 130
    else: return 400 + (pm - 250) * 100 / 130

# ----------------------------
# Forecast endpoint
# ----------------------------
@app.post("/forecast")
def forecast(request: PM25Request):
    pm_hist = np.array(request.pm25_history, dtype=np.float32).reshape(-1, 1)

    if len(pm_hist) != LOOKBACK:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {LOOKBACK} values, got {len(pm_hist)}"
        )

    # --- Step 1: log1p transform ---
    pm_log = np.log1p(pm_hist)

    # --- Step 2: residual scaling ---
    pm_scaled = scaler.transform(pm_log).reshape(1, -1)
    x = torch.tensor(pm_scaled).to(device)

    # --- Step 3: model prediction ---
    with torch.no_grad():
        _, forecast_out = model(x)

    forecast_out = forecast_out.cpu().numpy()
    if forecast_out.ndim == 3:
        forecast_out = forecast_out[0, 0, :]
    elif forecast_out.ndim == 2:
        forecast_out = forecast_out[0]
    forecast_out = forecast_out.reshape(-1, 1)

    # --- Step 4: inverse scale ---
    pm_pred_log = scaler.inverse_transform(forecast_out).flatten()

    # --- Step 5: inverse log ---
    pm_pred = np.expm1(pm_pred_log)

    # --- Step 6: clip extremes ---
    pm_pred = np.clip(pm_pred, 0, 500)

    # --- Step 7: convert to AQI ---
    aqi_pred = [pm25_to_aqi(v) for v in pm_pred]

    return {
        "pm25_forecast": [float(round(v, 1)) for v in pm_pred],
        "aqi_forecast": [float(round(v, 1)) for v in aqi_pred]
    }

# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=10000,
        reload=True
    )
