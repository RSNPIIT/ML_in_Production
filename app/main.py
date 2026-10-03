import os
import time
import logging
import joblib
import numpy as np
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field

# Configure Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
)
logger = logging.getLogger("ml_production_api")

app = FastAPI(
    title="ML in Production - Iris Classifier API",
    description="Production-ready FastAPI service for Iris species prediction with logging & monitoring metrics.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global Metrics Storage
metrics: Dict[str, Any] = {
    "total_requests": 0,
    "total_predictions": 0,
    "failed_predictions": 0,
    "total_latency_seconds": 0.0,
    "start_time": time.time()
}

CLASS_NAMES = ["setosa", "versicolor", "virginica"]
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")

model = None
scaler = None

@app.on_event("startup")
def load_artifacts():
    global model, scaler
    logger.info("Initializing ML models and scalers...")
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            logger.info(f"✅ Successfully loaded model from {MODEL_PATH} and scaler from {SCALER_PATH}")
        else:
            logger.error(f"❌ Artifact paths missing: {MODEL_PATH}, {SCALER_PATH}")
    except Exception as e:
        logger.error(f"❌ Failed to load artifacts: {str(e)}")

# Request Latency Middleware
@app.middleware("http")
async def track_latency(request: Request, call_next):
    metrics["total_requests"] += 1
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    metrics["total_latency_seconds"] += process_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    return response

# Pydantic Schemas
class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm (e.g. 5.1)", example=5.1)
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm (e.g. 3.5)", example=3.5)
    petal_length: float = Field(..., gt=0, description="Petal length in cm (e.g. 1.4)", example=1.4)
    petal_width: float = Field(..., gt=0, description="Petal width in cm (e.g. 0.2)", example=0.2)

class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    probabilities: Dict[str, float]
    latency_ms: float

@app.get("/", summary="Root Endpoint")
def root():
    return {
        "message": "Welcome to ML in Production API",
        "status": "online",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }

@app.get("/health", summary="Health Check")
def health_check():
    model_loaded = model is not None and scaler is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "uptime_seconds": round(time.time() - metrics["start_time"], 2)
    }

@app.get("/metrics", summary="Logging & Monitoring Metrics")
def get_metrics():
    total_preds = metrics["total_predictions"]
    avg_latency = (
        (metrics["total_latency_seconds"] / metrics["total_requests"]) * 1000
        if metrics["total_requests"] > 0
        else 0.0
    )
    return {
        "total_api_requests": metrics["total_requests"],
        "total_predictions": total_preds,
        "failed_predictions": metrics["failed_predictions"],
        "average_request_latency_ms": round(avg_latency, 2),
        "uptime_seconds": round(time.time() - metrics["start_time"], 2)
    }

@app.post("/predict", response_model=PredictionResponse, summary="Predict Iris Species")
def predict(data: IrisInput):
    if model is None or scaler is None:
        metrics["failed_predictions"] += 1
        logger.error("Prediction attempt failed: Model/Scaler not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded on server."
        )

    start = time.time()
    try:
        features = np.array([[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]])

        scaled_features = scaler.transform(features)
        pred_class = int(model.predict(scaled_features)[0])
        probabilities = model.predict_proba(scaled_features)[0].tolist()

        class_name = CLASS_NAMES[pred_class] if pred_class < len(CLASS_NAMES) else f"class_{pred_class}"
        prob_dict = {CLASS_NAMES[i]: round(float(probabilities[i]), 4) for i in range(len(CLASS_NAMES))}

        latency_ms = round((time.time() - start) * 1000, 2)
        metrics["total_predictions"] += 1

        logger.info(
            f"Prediction Success | Input: [{data.sepal_length}, {data.sepal_width}, {data.petal_length}, {data.petal_width}] "
            f"-> Pred: {class_name} ({pred_class}) | Latency: {latency_ms}ms"
        )

        return PredictionResponse(
            class_id=pred_class,
            class_name=class_name,
            probabilities=prob_dict,
            latency_ms=latency_ms
        )

    except Exception as e:
        metrics["failed_predictions"] += 1
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction processing error: {str(e)}"
        )
