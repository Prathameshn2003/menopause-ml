from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import os

# ---------------------------------
# APP INIT
# ---------------------------------
app = FastAPI(
    title="Menopause ML API",
    version="1.0",
    description="Machine Learning API for Menopause Stage Prediction"
)

# ---------------------------------
# CORS CONFIG
# ---------------------------------
# In production, replace * with your Vercel domain
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "*"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------
# LOAD MODEL & TOOLS (SAFE)
# ---------------------------------
try:
    with open("models/rf_model.pkl", "rb") as f:
        rf = pickle.load(f)

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("models/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

# ---------------------------------
# REQUEST SCHEMA
# ---------------------------------
class MenopauseInput(BaseModel):
    age: int
    estrogen: float
    fsh: float
    years_since_last_period: float
    irregular_periods: int
    missed_periods: int
    hot_flashes: int
    night_sweats: int
    sleep_problems: int
    vaginal_dryness: int
    joint_pain: int

# ---------------------------------
# RESPONSE SCHEMA (OPTIONAL BUT CLEAN)
# ---------------------------------
class MenopauseResponse(BaseModel):
    stage: str
    confidence: float
    probabilities: dict

# ---------------------------------
# HEALTH CHECK
# ---------------------------------
@app.get("/")
def health_check():
    return {"status": "Menopause ML API running"}

# ---------------------------------
# PREDICTION ENDPOINT
# ---------------------------------
@app.post("/predict-menopause", response_model=MenopauseResponse)
def predict_menopause(data: MenopauseInput):
    try:
        input_vector = np.array([[
            data.age,
            data.estrogen,
            data.fsh,
            data.years_since_last_period,
            data.irregular_periods,
            data.missed_periods,
            data.hot_flashes,
            data.night_sweats,
            data.sleep_problems,
            data.vaginal_dryness,
            data.joint_pain
        ]])

        X_scaled = scaler.transform(input_vector)
        probs = rf.predict_proba(X_scaled)[0]

        stage_index = int(np.argmax(probs))
        stage = le.inverse_transform([stage_index])[0]
        confidence = round(float(probs[stage_index]) * 100, 2)

        return {
            "stage": stage,
            "confidence": confidence,
            "probabilities": {
                cls: round(float(p) * 100, 2)
                for cls, p in zip(le.classes_, probs)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
