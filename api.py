"""
Air Quality Prediction API
===========================
Run with:
    uvicorn api:app --reload --port 8000

Then open: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, numpy as np, os

# ── App setup ──────────────────────────────────────────────
app = FastAPI(title="Air Quality API", version="1.0.0")

# Allow the Streamlit frontend to call this API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Load saved models ──────────────────────────────────────
MODELS_DIR = "models"

def load_model(filename):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise RuntimeError(f"File not found: {path} — run the notebook first.")
    return joblib.load(path)

try:
    regressor  = load_model("regressor.pkl")     # predicts AQI value (number)
    classifier = load_model("classifier.pkl")    # predicts risk level (Safe/Moderate/Dangerous)
    kmeans     = load_model("kmeans.pkl")        # assigns a pollution cluster
    scaler     = load_model("scaler.pkl")        # scales input features
    le         = load_model("label_encoder.pkl") # converts numbers back to class names
    feat_cols  = load_model("feature_cols.pkl")  # the order of features the models expect
    ready = True
except Exception as e:
    ready = False
    err_msg = str(e)

# ── Input schema ───────────────────────────────────────────
class AirInput(BaseModel):
    co_aqi:    float  # Carbon Monoxide AQI reading
    ozone_aqi: float  # Ozone AQI reading
    no2_aqi:   float  # Nitrogen Dioxide AQI reading
    pm25_aqi:  float  # Fine Particulate Matter AQI reading

    model_config = {
        "json_schema_extra": {
            "example": {
                "co_aqi": 2.0, "ozone_aqi": 45.0,
                "no2_aqi": 12.0, "pm25_aqi": 68.0
            }
        }
    }

# ── Helper functions ───────────────────────────────────────

def prepare_input(data: AirInput):
    """
    Build the feature array in the same column order used during training.
    Then scale it using the saved StandardScaler.
    """
    mapping = {
        "CO AQI Value":    data.co_aqi,
        "Ozone AQI Value": data.ozone_aqi,
        "NO2 AQI Value":   data.no2_aqi,
        "PM2.5 AQI Value": data.pm25_aqi,
    }
    raw = np.array([mapping[col] for col in feat_cols]).reshape(1, -1)
    return scaler.transform(raw)

def aqi_label(aqi: float) -> str:
    """Return a human-readable label for a given AQI value."""
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Moderate"
    if aqi <= 150:  return "Unhealthy for Sensitive Groups"
    if aqi <= 200:  return "Unhealthy"
    if aqi <= 300:  return "Very Unhealthy"
    return "Hazardous"

def check_ready():
    """Raise a 503 error if models failed to load."""
    if not ready:
        raise HTTPException(status_code=503, detail=f"Models not loaded. {err_msg}")

# ── Endpoints ──────────────────────────────────────────────

@app.get("/")
def home():
    """Quick status check — confirms the API is running."""
    return {"status": "running", "models_loaded": ready}


@app.get("/health")
def health():
    """Health check endpoint used by the Streamlit sidebar."""
    return {"ok": True, "models_loaded": ready}


@app.post("/predict/aqi")
def predict_aqi(data: AirInput):
    """
    Regression endpoint.
    Returns the predicted numeric AQI value for the given pollutant readings.
    """
    check_ready()
    X = prepare_input(data)
    value = float(regressor.predict(X)[0])
    return {
        "aqi_value":   round(value, 2),
        "description": aqi_label(value)
    }


@app.post("/predict/risk")
def predict_risk(data: AirInput):
    """
    Classification endpoint.
    Returns the risk level: Safe, Moderate, or Dangerous.
    """
    check_ready()
    X     = prepare_input(data)
    idx   = classifier.predict(X)[0]
    label = le.inverse_transform([idx])[0]
    colors = {"Safe": "#2ECC71", "Moderate": "#F39C12", "Dangerous": "#E74C3C"}
    return {
        "risk_level": label,
        "color":      colors.get(label, "#aaa")
    }


@app.post("/predict/cluster")
def predict_cluster(data: AirInput):
    """
    Clustering endpoint.
    Assigns the reading to one of the K pollution pattern groups.
    """
    check_ready()
    X = prepare_input(data)
    return {"cluster_id": int(kmeans.predict(X)[0])}


@app.post("/predict/all")
def predict_all(data: AirInput):
    """
    Main endpoint — runs all three models at once.
    This is what the Streamlit dashboard calls.
    """
    check_ready()
    X = prepare_input(data)

    aqi_value = float(regressor.predict(X)[0])
    risk_idx  = classifier.predict(X)[0]
    risk      = le.inverse_transform([risk_idx])[0]
    cluster   = int(kmeans.predict(X)[0])
    colors    = {"Safe": "#2ECC71", "Moderate": "#F39C12", "Dangerous": "#E74C3C"}

    return {
        "aqi_value":   round(aqi_value, 2),
        "description": aqi_label(aqi_value),
        "risk_level":  risk,
        "color":       colors.get(risk, "#aaa"),
        "cluster_id":  cluster,
    }
