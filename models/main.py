"""
main.py  (models/main.py)
--------------------------
Core prediction engine:
  - Loads trained XGBoost model + scaler
  - Runs feature engineering on incoming transaction
  - Returns fraud probability, decision, risk level
  - Generates human-readable fraud explanations
"""

import os
import joblib
import numpy as np
import xgboost as xgb

import pandas as pd
from models.features import engineer_features, FEATURE_COLS


# ── Paths ────────────────────────────────────────────────────────
_DIR         = os.path.dirname(__file__)
MODEL_PATH   = os.path.join(_DIR, "xgb_fraud.json")
SCALER_PATH  = os.path.join(_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(_DIR, "feature_names.pkl")
THRESHOLD_PATH = os.path.join(_DIR, "threshold.pkl")

# ── Lazy-load artefacts (loaded once at startup) ─────────────────
_model   = None
_scaler  = None
_features = None
_threshold = 0.5

def _load():
    global _model, _scaler, _features, _threshold
    if _model is None:
        _model = xgb.XGBClassifier()
        _model.load_model(MODEL_PATH)
        _scaler   = joblib.load(SCALER_PATH)
        _features = joblib.load(FEATURES_PATH)
        if os.path.exists(THRESHOLD_PATH):
            _threshold = joblib.load(THRESHOLD_PATH)


def _engineer(txn: dict):
    df = pd.DataFrame([txn])
    df = engineer_features(df)
    return df[FEATURE_COLS].values


def _risk_level(prob: float) -> str:
    if prob < 0.3:  return "LOW"
    if prob < 0.6:  return "MEDIUM"
    if prob < 0.85: return "HIGH"
    return "CRITICAL"


def _generate_summary(transaction, fraud_prob, risk_level, is_fraud, risk_factors):
    """
    Builds the human-readable summary. Deliberately keyed off `risk_level` /
    `risk_factors` rather than the binary `is_fraud` flag alone, so the
    wording can never contradict the risk badge or the indicators list --
    even for transactions that sit just below the flagging threshold.
    """
    amount = transaction.get("amount", 0)
    txn_type = transaction.get("type", "UNKNOWN")

    has_real_factors = bool(risk_factors) and risk_factors != ["No major fraud indicators detected"]
    is_elevated = risk_level in ("HIGH", "CRITICAL") or has_real_factors

    if is_fraud:
        return (
            f"This {txn_type} transaction of ₹{amount:,.2f} has been flagged as "
            f"{risk_level} risk with a fraud probability of {fraud_prob*100:.2f}%. "
            f"Key risk indicators include: "
            f"{', '.join(risk_factors) if has_real_factors else 'a fraud probability above the flagging threshold'}."
        )

    if is_elevated:
        return (
            f"This {txn_type} transaction of ₹{amount:,.2f} was not automatically flagged, "
            f"but carries a {risk_level.lower()} risk profile with a fraud probability of "
            f"{fraud_prob*100:.2f}%. Risk indicators present: "
            f"{', '.join(risk_factors)}. Recommended for manual review."
        )

    return (
        f"This {txn_type} transaction of ₹{amount:,.2f} appears legitimate "
        f"with a fraud probability of {fraud_prob*100:.2f}%. "
        f"No significant fraud indicators were detected."
    )


def predict_fraud(transaction: dict) -> dict:
    """
    Main prediction function.

    Parameters
    ----------
    transaction : dict  with keys matching the Transaction schema

    Returns
    -------
    dict with fraud_prob, is_fraud, risk_level, summary
    """
    _load()
    txn_dict = transaction

    features_scaled = _scaler.transform(_engineer(txn_dict))
    fraud_prob = float(_model.predict_proba(features_scaled)[0, 1])
    is_fraud = fraud_prob >= _threshold
    risk_level = _risk_level(fraud_prob)
    top_risk_factors = []

    amount = txn_dict.get("amount", 0)
    old_sender = txn_dict.get("oldbalanceOrg", 0)
    new_sender = txn_dict.get("newbalanceOrig", 0)
    old_receiver = txn_dict.get("oldbalanceDest", 0)
    txn_type = txn_dict.get("type", "")

    if amount > 50000:
        top_risk_factors.append("Large transaction amount")

    if txn_type in ["TRANSFER", "CASH_OUT"]:
        top_risk_factors.append(f"High-risk transaction type ({txn_type})")

    if old_sender > 0 and new_sender == 0:
        top_risk_factors.append("Sender account fully drained")

    if old_receiver == 0:
        top_risk_factors.append("Destination account has zero previous balance")

    if not top_risk_factors:
        top_risk_factors.append("No major fraud indicators detected")

    summary = _generate_summary(
        txn_dict,
        fraud_prob,
        risk_level,
        is_fraud,
        top_risk_factors,
    )

    # Confidence = how sure the model is in whichever verdict it gave.
    # (Previously this always echoed fraud_probability, which made a
    # "LEGITIMATE" verdict show a high "confidence" that actually measured
    # confidence in fraud — contradicting the label.)
    confidence = round((fraud_prob if is_fraud else (1 - fraud_prob)) * 100, 2)

    return {
        "fraud_probability": round(fraud_prob, 4),
        "confidence": f"{confidence}%",
        "threshold": f"{_threshold*100:.0f}%",
        "is_fraud": is_fraud,
        "risk_level": risk_level,
        "model": "XGBoost Fraud Classifier v1.0",
        "top_risk_factors": top_risk_factors,
        "summary": summary,
    }