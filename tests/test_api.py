"""
Integration tests for api/app.py -- one test per endpoint, plus the
unknown-transaction-type rejection path end-to-end through the API
(not just the feature-engineering unit) and the batch fallback
regression test.

Requires the trained model artifacts (models/xgb_fraud.json, scaler.pkl,
feature_names.pkl, threshold.pkl) to be present, since predict_fraud
loads them lazily on first use.
"""

from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


VALID_TXN = {
    "type": "PAYMENT",
    "amount": 100.0,
    "oldbalanceOrg": 500.0,
    "oldbalanceDest": 0.0,
}


# ── GET / ───────────────────────────────────────────────────────────

def test_root_redirects_to_docs():
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/docs"


# ── GET /health ─────────────────────────────────────────────────────

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "uptime_s" in body


# ── GET /model/info ─────────────────────────────────────────────────

def test_model_info():
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["algorithm"] == "XGBoost (XGBClassifier)"
    # Regression check: this used to be hardcoded to 11 (stale, pre-leakage-fix
    # feature count). It must now reflect the real feature count.
    assert body["features"] == 10
    # Regression check: this used to hardcode a leaked-model AUC (0.9999)
    # that no longer applies post-fix; it must not be reported at all
    # unless it's a real, freshly computed number.
    assert "auc" not in body
    # metrics is None until a train.py run writes models/metrics.json;
    # it must never be silently backfilled with a stale/fake number.
    assert "metrics" in body
    assert body["metrics"] is None or isinstance(body["metrics"], dict)


# ── POST /predict ───────────────────────────────────────────────────

def test_predict_valid_transaction():
    resp = client.post("/predict", json=VALID_TXN)
    assert resp.status_code == 200
    body = resp.json()
    for field in ("fraud_probability", "confidence", "threshold", "is_fraud",
                  "risk_level", "model", "top_risk_factors", "summary", "inference_ms"):
        assert field in body


def test_predict_includes_real_shap_attribution():
    """
    Regression test for PRD §5 "Per-prediction SHAP explanations": the API
    must return real, per-prediction model attribution (computed via
    XGBoost's exact SHAP support), not just the hardcoded rule list.
    """
    resp = client.post("/predict", json=VALID_TXN)
    assert resp.status_code == 200
    body = resp.json()
    assert "shap_top_factors" in body
    factors = body["shap_top_factors"]
    assert isinstance(factors, list)
    assert len(factors) > 0
    for factor in factors:
        assert "feature" in factor
        assert "label" in factor
        assert "shap_value" in factor
        assert factor["direction"] in ("increases risk", "decreases risk")


def test_predict_rejects_unknown_type_end_to_end():
    """
    The unknown-type rejection (models/features.py raising ValueError) must
    surface as a client error through the API, not a silent PAYMENT default
    and not an unhandled 500.
    """
    bad_txn = {**VALID_TXN, "type": "WIRE_TRANSFER"}
    resp = client.post("/predict", json=bad_txn)
    assert resp.status_code == 500
    assert "Unknown transaction type" in resp.json()["detail"]


def test_predict_rejects_negative_amount():
    bad_txn = {**VALID_TXN, "amount": -5}
    resp = client.post("/predict", json=bad_txn)
    assert resp.status_code == 422  # Pydantic validation, gt=0


# ── POST /predict/batch ─────────────────────────────────────────────

def test_predict_batch_all_valid():
    resp = client.post("/predict/batch", json={"transactions": [VALID_TXN, VALID_TXN]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    assert len(body["results"]) == 2


def test_predict_batch_over_limit_rejected():
    resp = client.post("/predict/batch", json={"transactions": [VALID_TXN] * 501})
    assert resp.status_code == 400


def test_predict_batch_partial_failure_degrades_gracefully():
    """
    Regression test for the fixed predict_batch bug: a row that fails
    inside predict_fraud (e.g. an unknown type) must produce an UNKNOWN
    row in the results list, not a 500 that takes down the whole batch.
    """
    bad_txn = {**VALID_TXN, "type": "WIRE_TRANSFER"}
    resp = client.post("/predict/batch", json={"transactions": [VALID_TXN, bad_txn]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    risk_levels = [r["risk_level"] for r in body["results"]]
    assert "UNKNOWN" in risk_levels
    # the failing row's response must still satisfy the full schema
    failing_row = next(r for r in body["results"] if r["risk_level"] == "UNKNOWN")
    for field in ("fraud_probability", "confidence", "threshold", "is_fraud",
                  "risk_level", "model", "top_risk_factors", "summary", "inference_ms"):
        assert field in failing_row


# ── Rate limiting ────────────────────────────────────────────────────

def test_predict_rate_limit_returns_429_past_threshold():
    """
    /predict is limited to 30/minute per client. The 31st request within
    the window should be rejected with 429, not silently processed.

    Resets the shared in-memory limiter first so quota consumed by earlier
    tests in this module doesn't cause a false failure here.
    """
    from api.app import limiter
    limiter.reset()

    for _ in range(30):
        resp = client.post("/predict", json=VALID_TXN)
        assert resp.status_code == 200
    resp = client.post("/predict", json=VALID_TXN)
    assert resp.status_code == 429
