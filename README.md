# Getting Started with ML in Production Workshop - Assignment

Production-grade Machine Learning deployment pipeline built with **Scikit-Learn**, **FastAPI**, **Docker**, **GitHub**, and **Render**.

---

## 📌 Architecture Overview

```
[ Model Training (train.py) ]
             │
             ▼
   [ Artifacts Export ] ──► (models/model.pkl & scaler.pkl)
             │
             ▼
  [ FastAPI Application ] ──► /predict, /metrics, /health, /docs
             │
             ▼
    [ Docker Container ] ──► Dockerfile (Python 3.11-slim)
             │
             ▼
    [ GitHub Repo ] ────► RSNPIIT/ML_in_Production
             │
             ▼
  [ Render Web Service ] ──► Live Deployment with Logging & Monitoring
```

---

## 🚀 Pipeline Components

1. **Model Training & Export (`train.py`)**
   - Trained a `RandomForestClassifier` on the Iris dataset.
   - Scaled features using `StandardScaler`.
   - Exported artifacts to `models/model.pkl` and `models/scaler.pkl`.

2. **FastAPI Application (`app/main.py`)**
   - Structured JSON logging.
   - Request latency middleware and metrics collector (`/metrics`).
   - Request validation using `Pydantic` schemas.
   - Endpoints:
     - `GET /`: API welcome and operational links.
     - `GET /health`: Service and model status.
     - `GET /metrics`: Aggregated prediction counts, errors, and average latency.
     - `POST /predict`: Input feature prediction endpoint.

3. **Containerization (`Dockerfile`)**
   - Light-weight `python:3.11-slim` base image.
   - Dynamic port binding for Render (`$PORT`).

---

## 🧪 API Usage & Testing

### 1. Predict Endpoint (`POST /predict`)

**Sample Request Payload:**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**cURL Command:**
```bash
curl -X POST "https://<your-render-app-url>/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "sepal_length": 5.1,
       "sepal_width": 3.5,
       "petal_length": 1.4,
       "petal_width": 0.2
     }'
```

**Sample Response:**
```json
{
  "class_id": 0,
  "class_name": "setosa",
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  },
  "latency_ms": 2.45
}
```

---

### 2. Metrics Endpoint (`GET /metrics`)

**cURL Command:**
```bash
curl -X GET "https://<your-render-app-url>/metrics"
```

**Sample Response:**
```json
{
  "total_api_requests": 15,
  "total_predictions": 12,
  "failed_predictions": 0,
  "average_request_latency_ms": 3.12,
  "uptime_seconds": 1240.5
}
```

---

## 🛠️ Deploying on Render

1. Log in to [Render Dashboard](https://dashboard.render.com).
2. Click **New +** -> **Web Service**.
3. Connect your repository: `RSNPIIT/ML_in_Production`.
4. Select **Docker** environment.
5. Click **Create Web Service**.
