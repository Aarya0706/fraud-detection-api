"""
test_api.py
-----------
Quick sanity-check tests — run AFTER training the model.

Usage:
  # Start the server first:
  uvicorn api.app:app --port 8000

  # Then in another terminal:
  python test_api.py
"""

import httpx
import json

BASE = "http://localhost:8000"

# ── Test cases ───────────────────────────────────────────────────
SUSPICIOUS_TXN = {
    "type":           "TRANSFER",
    "amount":         9823.50,
    "oldbalanceOrg":  10000.0,
    "newbalanceOrig": 176.50,       # account nearly drained
    "oldbalanceDest": 0.0,          # new destination account
    "newbalanceDest": 9823.50,
    "recency_hours":  0.5,          # very recent last transaction
    "txn_count_24h":  12,           # high frequency
    "is_dest_new":    1,            # new destination
}

LEGIT_TXN = {
    "type":           "PAYMENT",
    "amount":         250.00,
    "oldbalanceOrg":  5000.0,
    "newbalanceOrig": 4750.0,
    "oldbalanceDest": 3000.0,
    "newbalanceDest": 3250.0,
    "recency_hours":  48.0,
    "txn_count_24h":  2,
    "is_dest_new":    0,
}


def test_health():
    r = httpx.get(f"{BASE}/health")
    assert r.status_code == 200
    print("✅  /health  →", r.json())


def test_model_info():
    r = httpx.get(f"{BASE}/model/info")
    assert r.status_code == 200
    print("✅  /model/info  →", r.json())


def test_predict_fraud():
    r = httpx.post(f"{BASE}/predict", json=SUSPICIOUS_TXN)
    assert r.status_code == 200
    data = r.json()
    print("\n🚨  Suspicious transaction result:")
    print(json.dumps(data, indent=2))


def test_predict_legit():
    r = httpx.post(f"{BASE}/predict", json=LEGIT_TXN)
    assert r.status_code == 200
    data = r.json()
    print("\n✅  Legitimate transaction result:")
    print(json.dumps(data, indent=2))


def test_batch():
    r = httpx.post(
        f"{BASE}/predict/batch",
        json={"transactions": [SUSPICIOUS_TXN, LEGIT_TXN, SUSPICIOUS_TXN]},
    )
    assert r.status_code == 200
    data = r.json()
    print(f"\n📦  Batch result: {data['fraud_count']}/{data['total']} flagged as fraud "
          f"in {data['elapsed_ms']} ms")


if __name__ == "__main__":
    test_health()
    test_model_info()
    test_predict_fraud()
    test_predict_legit()
    test_batch()
    print("\n✅  All tests passed!")
