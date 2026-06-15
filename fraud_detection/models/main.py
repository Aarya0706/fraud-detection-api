"""
main.py  (models/main.py)
--------------------------
Core prediction engine:
  - Loads trained XGBoost model + scaler
  - Runs feature engineering on incoming transaction
  - Returns fraud probability, decision, risk level
  - Calls LLM (OpenAI GPT-3.5) for human-readable explanation
"""

import os
import joblib
import numpy as np
import xgboost as xgb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────
_DIR         = os.path.dirname(__file__)
MODEL_PATH   = os.path.join(_DIR, "xgb_fraud.json")
SCALER_PATH  = os.path.join(_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(_DIR, "feature_names.pkl")

# ── Lazy-load artefacts (loaded once at startup) ─────────────────
_model   = None
_scaler  = None
_features = None

def _load():
    global _model, _scaler, _features
    if _model is None:
        _model = xgb.XGBClassifier()
        _model.load_model(MODEL_PATH)
        _scaler   = joblib.load(SCALER_PATH)
        _features = joblib.load(FEATURES_PATH)

# ── LLM client ───────────────────────────────────────────────────
_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _llm


TYPE_MAP = {"PAYMENT": 0, "TRANSFER": 1, "CASH_OUT": 2, "DEBIT": 3, "CASH_IN": 4}


def _engineer(txn: dict) -> np.ndarray:
    """Turn a raw transaction dict into the feature vector expected by the model."""
    import math

    t = txn
    type_enc           = TYPE_MAP.get(t.get("type", "PAYMENT"), 0)
    amount             = float(t.get("amount", 0))
    old_bal_orig       = float(t.get("oldbalanceOrg", 0))
    new_bal_orig       = float(t.get("newbalanceOrig", 0))
    old_bal_dest       = float(t.get("oldbalanceDest", 0))
    new_bal_dest       = float(t.get("newbalanceDest", 0))
    recency_hours      = float(t.get("recency_hours", 24))
    txn_count_24h      = float(t.get("txn_count_24h", 1))
    is_dest_new        = float(t.get("is_dest_new", 0))

    balance_change_orig   = new_bal_orig - old_bal_orig
    balance_change_dest   = new_bal_dest - old_bal_dest
    orig_drained          = 1 if (old_bal_orig > 0 and new_bal_orig == 0) else 0
    amount_ratio_orig     = amount / (old_bal_orig + 1) if old_bal_orig > 0 else 0
    dest_balance_anomaly  = 1 if old_bal_dest == 0 else 0

    log_amount       = math.log1p(amount)
    log_old_orig     = math.log1p(old_bal_orig)
    log_new_orig     = math.log1p(new_bal_orig)
    log_old_dest     = math.log1p(old_bal_dest)
    log_new_dest     = math.log1p(new_bal_dest)

    return np.array([[
        type_enc, log_amount, log_old_orig, log_new_orig,
        log_old_dest, log_new_dest,
        balance_change_orig, balance_change_dest,
        orig_drained, amount_ratio_orig, dest_balance_anomaly,
        recency_hours, txn_count_24h, is_dest_new,
    ]])


def _risk_level(prob: float) -> str:
    if prob < 0.3:  return "LOW"
    if prob < 0.6:  return "MEDIUM"
    if prob < 0.85: return "HIGH"
    return "CRITICAL"


def predict_fraud(transaction: dict) -> dict:
    """
    Main prediction function.

    Parameters
    ----------
    transaction : dict  with keys matching the Transaction schema

    Returns
    -------
    dict with fraud_prob, is_fraud, risk_level, summary (from LLM)
    """
    _load()
    txn_dict = transaction

    features_scaled = _scaler.transform(_engineer(txn_dict))
    fraud_prob = float(_model.predict_proba(features_scaled)[0, 1])
    is_fraud   = fraud_prob >= 0.5
    risk_level = _risk_level(fraud_prob)

    # ── LLM explanation ───────────────────────────────────────────
    txn_type = txn_dict.get("type", "UNKNOWN")
    prompt = f"""You are a fraud detection analyst. Analyze this transaction and provide a clear, professional summary.

ANALYSIS RESULTS:
- Fraud Probability: {fraud_prob*100:.2f}%
- Decision: {"FRAUD DETECTED" if is_fraud else "LEGITIMATE TRANSACTION"}
- Risk Level: {risk_level}

TRANSACTION DETAILS:
- Amount: ${txn_dict.get('amount', 0):,.2f}
- Type: {txn_type}
- Account Balance Before: ${txn_dict.get('oldbalanceOrg', 0):,.2f}
- Account Balance After: ${txn_dict.get('newbalanceOrig', 0):,.2f}
- Destination Balance Before: ${txn_dict.get('oldbalanceDest', 0):,.2f}
- Destination Balance After: ${txn_dict.get('newbalanceDest', 0):,.2f}
- New Destination: {"Yes" if txn_dict.get('is_dest_new') == 1 else "No"}
- Hours Since Last Transaction: {txn_dict.get('recency_hours', 0):.1f}
- Transactions in Last 24h: {txn_dict.get('txn_count_24h', 0)}

Provide a 2-3 sentence professional summary explaining why this transaction is \
{"flagged as fraud" if is_fraud else "considered legitimate"}. Focus on the key risk indicators."""

    if is_fraud:
       summary = (
        f"Fraud Alert. This transaction has been flagged as suspicious. "
        f"Risk level: {risk_level}. "
        f"Fraud probability: {fraud_prob*100:.1f}%."
    )
    else:
       summary = (
        f"Transaction appears legitimate. "
        f"Risk level: {risk_level}. "
        f"Fraud probability: {fraud_prob*100:.1f}%."
    )

    return {
        "fraud_probability": round(fraud_prob, 4),
        "is_fraud":          is_fraud,
        "risk_level":        risk_level,
        "summary":           summary,
    }
