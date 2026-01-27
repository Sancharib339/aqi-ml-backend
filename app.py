from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import numpy as np

from nbeats_pytorch.model import NBeatsNet
from sklearn.preprocessing import StandardScaler
from torch.serialization import add_safe_globals

add_safe_globals([StandardScaler])

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(
    "model/nbeats_pm25.pt",
    map_location=device,
    weights_only=False
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


class PM25Request(BaseModel):
    pm25_history: List[float]


def pm25_to_aqi(pm):
    if pm <= 30: return (pm / 30) * 50
    elif pm <= 60: return 50 + (pm - 30) * 50 / 30
    elif pm <= 90: return 100 + (pm - 60) * 100 / 30
    elif pm <= 120: return 200 + (pm - 90) * 100 / 30
    elif pm <= 250: return 300 + (pm - 120) * 100 / 130
    else: return 400 + (pm - 250) * 100 / 130


@app.post("/forecast")
def forecast(request: PM25Request):

    pm_hist = np.array(request.pm25_history).reshape(-1, 1)

    if len(pm_hist) != LOOKBACK:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {LOOKBACK} values, got {len(pm_hist)}"
        )

    pm_scaled = scaler.transform(pm_hist)
    pm_scaled = pm_scaled.reshape(1, -1)

    x = torch.tensor(pm_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        _, forecast = model(x)

    pm_pred = scaler.inverse_transform(
        forecast.cpu().numpy().reshape(-1, 1)
    ).flatten()

    aqi_pred = [pm25_to_aqi(v) for v in pm_pred]

    return {
        "pm25_forecast": pm_pred.tolist(),
        "aqi_forecast": aqi_pred
    }
