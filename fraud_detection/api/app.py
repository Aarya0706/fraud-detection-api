"""
app.py  (api/app.py)
---------------------
FastAPI REST API for fraud detection.
Exposes:
  POST /predict          — single transaction prediction
  POST /predict/batch    — batch predictions
  GET  /health           — health check
  GET  /model/info       — model metadata

Run:
  uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""

import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# import prediction engine
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.main import predict_fraud

# ─────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────

class Transaction(BaseModel):
    type:            str   = Field(..., example="TRANSFER",
                                   description="PAYMENT | TRANSFER | CASH_OUT | DEBIT | CASH_IN")
    amount:          float = Field(..., gt=0, example=9823.50)
    oldbalanceOrg:   float = Field(..., ge=0, example=10000.0)
    newbalanceOrig:  float = Field(..., ge=0, example=176.50)
    oldbalanceDest:  float = Field(..., ge=0, example=0.0)
    newbalanceDest:  float = Field(..., ge=0, example=9823.50)
    recency_hours:   float = Field(24.0, ge=0, example=1.5,
                                   description="Hours since sender's last transaction")
    txn_count_24h:   int   = Field(1,    ge=0, example=8,
                                   description="Number of transactions by sender in last 24h")
    is_dest_new:     int   = Field(0,    ge=0, le=1, example=1,
                                   description="1 if destination account is new/unseen")


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud:          bool
    risk_level:        str
    summary:           str
    inference_ms:      float


class BatchRequest(BaseModel):
    transactions: List[Transaction]


class BatchResponse(BaseModel):
    results:     List[PredictionResponse]
    total:       int
    fraud_count: int
    elapsed_ms:  float


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

app = FastAPI(
    title="AI-Driven Financial Fraud Detection API",
    description=(
    "XGBoost-powered REST API for real-time fraud detection."
    ),
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_startup_time = time.time()


@app.get("/health", tags=["System"])
def health():
    return {
        "status":      "ok",
        "uptime_s":    round(time.time() - _startup_time, 1),
        "model":       "XGBoost + GPT-3.5",
        "version":     "1.0.0",
    }


@app.get("/model/info", tags=["System"])
def model_info():
    return {
        "algorithm":       "XGBoost (XGBClassifier)",
        "training_rows":   "6,300,000",
        "auc":             0.9998,
        "precision":       "80%",
        "recall":          "99%",
        "llm_explainer":   "GPT-3.5-turbo",
        "features":        14,
        "target_latency":  "<100 ms",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(txn: Transaction):
    """
    Predict fraud for a single financial transaction.
    Returns fraud probability, decision, risk level, and LLM-generated explanation.
    """
    t0 = time.time()
    try:
        result = predict_fraud(txn.model_dump())
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run `python models/train.py` first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    inference_ms = round((time.time() - t0) * 1000, 2)
    return PredictionResponse(**result, inference_ms=inference_ms)


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(req: BatchRequest):
    """
    Predict fraud for a batch of transactions (max 500).
    """
    if len(req.transactions) > 500:
        raise HTTPException(status_code=400, detail="Max 500 transactions per batch.")

    t0 = time.time()
    results = []
    for txn in req.transactions:
        t_start = time.time()
        try:
            r = predict_fraud(txn.model_dump())
        except Exception as e:
            r = {
                "fraud_probability": 0.0,
                "is_fraud": False,
                "risk_level": "UNKNOWN",
                "summary": f"Error: {str(e)}",
            }
        results.append(PredictionResponse(**r, inference_ms=round((time.time()-t_start)*1000, 2)))

    fraud_count = sum(1 for r in results if r.is_fraud)
    return BatchResponse(
        results=results,
        total=len(results),
        fraud_count=fraud_count,
        elapsed_ms=round((time.time()-t0)*1000, 2),
    )
