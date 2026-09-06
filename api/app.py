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

import json
import time
from typing import List

from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# import prediction engine
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.main import predict_fraud
from models.features import FEATURE_COLS

METRICS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "metrics.json")


def _load_metrics():
    """
    Returns the metrics train.py saved on its last run, or None if a
    training run hasn't produced models/metrics.json yet (e.g. this is
    still the pre-fix model bundle). See PRD 3.2.
    """
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

# ─────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────

class Transaction(BaseModel):
    # NOTE: no newbalanceOrig / newbalanceDest here on purpose -- those are
    # POST-transaction balances that don't exist yet at authorization time,
    # and including them let the old model leak the label instead of
    # learning real fraud signal. Only pre-transaction fields below.
    type:            str   = Field(..., example="TRANSFER",
                                   description="PAYMENT | TRANSFER | CASH_OUT | DEBIT | CASH_IN")
    amount:          float = Field(..., gt=0, example=9823.50)
    oldbalanceOrg:   float = Field(..., ge=0, example=10000.0)
    oldbalanceDest:  float = Field(..., ge=0, example=0.0)
    recency_hours:   float = Field(24.0, ge=0, example=1.5,
                                   description="Hours since sender's last transaction")
    txn_count_24h:   int   = Field(1,    ge=0, example=8,
                                   description="Number of transactions by sender in last 24h")
    is_dest_new:     int   = Field(0,    ge=0, le=1, example=1,
                                   description="1 if destination account is new/unseen")


class PredictionResponse(BaseModel):
    fraud_probability: float
    confidence: str
    threshold: str
    is_fraud: bool
    risk_level: str
    model: str
    top_risk_factors: List[str]
    summary: str
    inference_ms: float


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

# ── CORS ─────────────────────────────────────────────────────────
# Known deployed frontends, plus common local dev ports. Override/extend via
# the ALLOWED_ORIGINS env var (comma-separated) without touching code, e.g.
# for a new frontend deployment or a staging URL.
_DEFAULT_ORIGINS = [
    "https://fraud-detection-api-eta.vercel.app",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]
_env_origins = os.environ.get("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = _DEFAULT_ORIGINS + [o.strip() for o in _env_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rate limiting ────────────────────────────────────────────────
# Keyed by client IP. Prediction routes get an explicit lower limit since
# they're the expensive/abusable ones; everything else falls back to the
# global default.
limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Optional API-key auth ───────────────────────────────────────
# Off by default (fine for a public portfolio demo). Set the API_KEY env
# var to require an `X-API-Key` header on prediction routes -- lets this
# be locked down without a code change if it's ever exposed beyond a demo.
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_REQUIRED_API_KEY = os.environ.get("API_KEY")


def require_api_key(key: str = Security(_api_key_header)):
    if _REQUIRED_API_KEY and key != _REQUIRED_API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")
    return True


_startup_time = time.time()


@app.get("/", include_in_schema=False)
def root():
    """Friendly landing route — sends visitors to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["System"])
def health():
    return {
        "status":      "ok",
        "uptime_s":    round(time.time() - _startup_time, 1),
        "model":       "XGBoost Fraud Classifier v1.0",
        "version":     "1.0.0",
    }


@app.get("/model/info", tags=["System"])
def model_info():
    metrics = _load_metrics()
    return {
        "algorithm":       "XGBoost (XGBClassifier)",
        "training_rows":   "6,362,620",
        "features":        len(FEATURE_COLS),
        "target_latency":  "<100 ms",
        # None until the next `python -m models.train` run writes
        # models/metrics.json (nothing trustworthy is hardcoded here --
        # the old hardcoded 0.9999 AUC was itself the bug, see PRD 3.2).
        "metrics":         metrics,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
@limiter.limit("30/minute")
def predict(request: Request, txn: Transaction, _auth: bool = Security(require_api_key)):
    """
    Predict fraud for a single financial transaction.
    Returns fraud probability, decision, risk level, and a rule-based
    explanation summary.
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
@limiter.limit("10/minute")
def predict_batch(request: Request, req: BatchRequest, _auth: bool = Security(require_api_key)):
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
            # Must populate every PredictionResponse field, or FastAPI raises
            # a 500 Pydantic validation error for this row and takes the
            # whole batch down with it -- defeating the point of a fallback.
            r = {
                "fraud_probability": 0.0,
                "confidence": "0%",
                "threshold": "N/A",
                "is_fraud": False,
                "risk_level": "UNKNOWN",
                "model": "N/A",
                "top_risk_factors": [],
                "summary": f"Error scoring this transaction: {str(e)}",
            }
        results.append(PredictionResponse(**r, inference_ms=round((time.time()-t_start)*1000, 2)))

    fraud_count = sum(1 for r in results if r.is_fraud)
    return BatchResponse(
        results=results,
        total=len(results),
        fraud_count=fraud_count,
        elapsed_ms=round((time.time()-t0)*1000, 2),
    )