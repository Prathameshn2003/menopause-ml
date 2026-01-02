from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI(
    title="Menopause ML API",
    version="1.0",
    description="Machine Learning API for Menopause Stage Prediction"
)

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LOAD MODELS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    rf = pickle.load(open(os.path.join(BASE_DIR, "models/rf_model.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(BASE_DIR, "models/scaler.pkl"), "rb"))
    le = pickle.load(open(os.path.join(BASE_DIR, "models/label_encoder.pkl"), "rb"))
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

# ---------------- SCHEMAS ----------------
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

class MenopauseResponse(BaseModel):
    stage: str
    confidence: float
    probabilities: dict

@app.get("/")
def health():
    return {"status": "Menopause ML API running"}

@app.post("/predict-menopause", response_model=MenopauseResponse)
def predict_menopause(data: MenopauseInput):
    try:
        X = np.array([[ 
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

        X_scaled = scaler.transform(X)
        probs = rf.predict_proba(X_scaled)[0]

        idx = int(np.argmax(probs))
        stage = le.inverse_transform([idx])[0]
        confidence = round(float(probs[idx]) * 100, 2)

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
