"""
Unit tests for models/monitoring.py.

log_prediction()/get_summary() share module-level state (by design --
it's meant to aggregate across the whole running process), so each test
resets that state first rather than relying on import order or test order.
"""

import json

import pytest

from models import monitoring


@pytest.fixture(autouse=True)
def _reset_monitoring_state(tmp_path, monkeypatch):
    """Gives every test a clean counter/window and its own scratch log file."""
    monkeypatch.setattr(monitoring, "_total_predictions", 0)
    monkeypatch.setattr(monitoring, "_total_fraud_flagged", 0)
    monkeypatch.setattr(monitoring, "_recent_latencies_ms", monitoring.deque(maxlen=monitoring._WINDOW))
    monkeypatch.setattr(monitoring, "_recent_probabilities", monitoring.deque(maxlen=monitoring._WINDOW))
    monkeypatch.setattr(monitoring, "_counts_by_model_version", {})
    monkeypatch.setattr(monitoring, "LOG_PATH", str(tmp_path / "predictions.jsonl"))
    yield


def test_log_prediction_updates_counts_and_fraud_rate():
    monitoring.log_prediction(
        endpoint="/predict", model_version="abc123",
        fraud_probability=0.9, risk_level="CRITICAL",
        is_fraud=True, inference_ms=12.5,
    )
    monitoring.log_prediction(
        endpoint="/predict", model_version="abc123",
        fraud_probability=0.1, risk_level="LOW",
        is_fraud=False, inference_ms=8.0,
    )

    summary = monitoring.get_summary()
    assert summary["total_predictions"] == 2
    assert summary["fraud_flagged"] == 1
    assert summary["fraud_rate"] == 0.5
    assert summary["predictions_by_model_version"] == {"abc123": 2}


def test_log_prediction_coerces_numpy_bool_and_writes_valid_json_line(tmp_path):
    """
    Regression test: predict_fraud's is_fraud is often a numpy.bool_ from a
    threshold comparison, which json.dumps cannot serialize on its own.
    """
    np = pytest.importorskip("numpy")
    monitoring.log_prediction(
        endpoint="/predict", model_version="abc123",
        fraud_probability=0.7, risk_level="HIGH",
        is_fraud=np.bool_(True), inference_ms=10.0,
    )

    with open(monitoring.LOG_PATH) as f:
        lines = [line for line in f if line.strip()]
    assert len(lines) == 1

    entry = json.loads(lines[0])  # raises if the numpy bool_ leaked through
    assert entry["is_fraud"] is True
    assert entry["model_version"] == "abc123"


def test_get_summary_with_no_traffic_returns_nones_not_errors():
    summary = monitoring.get_summary()
    assert summary["total_predictions"] == 0
    assert summary["fraud_rate"] is None
    assert summary["latency_ms"]["mean"] is None
    assert summary["fraud_probability_drift"]["stdev"] is None


def test_percentile_matches_known_values():
    values = [10, 20, 30, 40, 50]
    assert monitoring._percentile(values, 50) == 30
    assert monitoring._percentile([], 95) is None