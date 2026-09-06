"""
monitoring.py  (models/monitoring.py)
--------------------------------------
Lightweight structured logging + in-process aggregation of prediction
traffic, so prediction volume, latency, and score drift can be observed
over time (PRD §5: "Structured logging & monitoring").

Every prediction (single or one row of a batch) is appended as one JSON
line to LOG_PATH, tagged with `model_version` so a metric shift can be
correlated back to a specific row in models/model_registry.jsonl.

Scope/limits (deliberate, for a portfolio-scale deployment):
  - In-memory aggregates reset on process restart. Render's free tier
    spins the container down after ~15 min idle, so GET /metrics/predictions
    reports "since this instance's last cold start", not lifetime totals.
  - The JSONL file lives on the container's local disk, which is
    ephemeral on Render's free tier -- it does NOT survive a redeploy.
    Fine for local debugging / a demo; ship it to external storage
    (S3, a log drain, a real APM) before relying on it for anything else.
  - Rolling windows are capped at _WINDOW entries so memory use doesn't
    grow unbounded with traffic.
"""

import json
import os
import statistics
import threading
import time
from collections import deque
from datetime import datetime, timezone

_DIR = os.path.dirname(os.path.dirname(__file__))
LOG_PATH = os.path.join(_DIR, "logs", "predictions.jsonl")
_WINDOW = 1000  # rolling window size for latency/drift stats

_lock = threading.Lock()
_started_at = time.time()
_total_predictions = 0
_total_fraud_flagged = 0
_recent_latencies_ms = deque(maxlen=_WINDOW)
_recent_probabilities = deque(maxlen=_WINDOW)
_counts_by_model_version = {}


def log_prediction(*, endpoint, model_version, fraud_probability, risk_level,
                    is_fraud, inference_ms):
    """
    Records one prediction: appends a JSON line to disk and updates the
    in-memory rolling aggregates used by get_summary(). Never raises --
    a logging failure should never take down a prediction response.
    """
    global _total_predictions, _total_fraud_flagged

    # Values arriving here can be numpy scalars (e.g. is_fraud is often a
    # numpy.bool_ from a threshold comparison) which json.dumps can't
    # serialize -- coerce to native Python types up front.
    is_fraud = bool(is_fraud)
    fraud_probability = float(fraud_probability) if fraud_probability is not None else None
    inference_ms = float(inference_ms) if inference_ms is not None else None

    entry = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "model_version": model_version,
        "fraud_probability": fraud_probability,
        "risk_level": risk_level,
        "is_fraud": is_fraud,
        "inference_ms": inference_ms,
    }

    with _lock:
        _total_predictions += 1
        if is_fraud:
            _total_fraud_flagged += 1
        _recent_latencies_ms.append(inference_ms)
        _recent_probabilities.append(fraud_probability)
        _counts_by_model_version[model_version] = (
            _counts_by_model_version.get(model_version, 0) + 1
        )

    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def _percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100)
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def get_summary():
    """
    Aggregate stats for GET /metrics/predictions: volume, fraud rate,
    latency distribution, and a simple score-drift signal (mean/stdev of
    fraud_probability over the recent window) -- a shift here can flag
    either a changing traffic mix or a model that needs retraining.
    """
    with _lock:
        recent_latencies = list(_recent_latencies_ms)
        recent_probs = list(_recent_probabilities)
        total = _total_predictions
        fraud_flagged = _total_fraud_flagged
        by_version = dict(_counts_by_model_version)

    return {
        "since_utc": datetime.fromtimestamp(_started_at, tz=timezone.utc).isoformat(),
        "total_predictions": total,
        "fraud_flagged": fraud_flagged,
        "fraud_rate": round(fraud_flagged / total, 4) if total else None,
        "latency_ms": {
            "window_size": len(recent_latencies),
            "mean": round(statistics.fmean(recent_latencies), 2) if recent_latencies else None,
            "p50": round(_percentile(recent_latencies, 50), 2) if recent_latencies else None,
            "p95": round(_percentile(recent_latencies, 95), 2) if recent_latencies else None,
            "p99": round(_percentile(recent_latencies, 99), 2) if recent_latencies else None,
        },
        "fraud_probability_drift": {
            "window_size": len(recent_probs),
            "mean": round(statistics.fmean(recent_probs), 4) if recent_probs else None,
            "stdev": round(statistics.pstdev(recent_probs), 4) if len(recent_probs) > 1 else None,
        },
        "predictions_by_model_version": by_version,
    }
