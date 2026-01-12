from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

# Define the input schema using Pydantic for validation
class AQIInput(BaseModel):
    PM2_5: float
    PM10: float
    NO2: float
    SO2: float
    CO: float
    Ozone: float
    Holidays_Count: int = 0  # default 0 if not provided
    Days: int = 4            # default 4 if not provided

app = FastAPI(title="Delhi AQI Predictor")

# Load your pickle model
with open("aqi_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the expected feature order
feature_cols = [
    "Holidays_Count",
    "Days",
    "PM2_5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "Ozone"
]

@app.get("/")
def home():
    return {"status": "AQI model is running"}

@app.post("/predict")
def predict_aqi(data: AQIInput):
    # Convert Pydantic model to dict
    input_data = data.dict()

    # Prepare features in the correct order
    try:
        features = np.array([[input_data[col] for col in feature_cols]])
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required feature: {e}")

    # Make prediction
    prediction = model.predict(features)[0]
    return {"AQI": float(prediction)}
