"""
Smoke test for models/train.py.

Not a model-quality test -- there's no assertion on ROC-AUC or any other
score, since a tiny synthetic sample can't produce a meaningful one. This
only exists to catch a *broken pipeline* (an exception, a crashed SMOTE
step, a save that silently fails) before it ships, per PRD's "Known
issues": train.py itself had zero test coverage.

Runs the real train() function end-to-end against a small hand-built
PaySim-shaped CSV, redirected into a temp model directory so it never
touches the real models/*.{json,pkl} artifacts.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from models import train as train_module


def _synthetic_paysim_df(n_legit=300, n_fraud=12, seed=0):
    # Fraud rate must stay well under train.py's SMOTE target of 10%
    # minority share, or SMOTE has nothing to oversample and raises.
    """
    Builds a small DataFrame with the same columns models.train.load_data
    reads from paysim.csv, with a real (if crude) fraud pattern baked in --
    fraud rows fully drain the sender -- so would_drain_orig has something
    to learn and SMOTE has a minority class to resample.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n_legit):
        balance = rng.uniform(100, 10_000)
        rows.append({
            "step": int(rng.integers(1, 500)),
            "type": rng.choice(["PAYMENT", "CASH_IN", "DEBIT"]),
            "amount": balance * rng.uniform(0.01, 0.3),
            "nameOrig": f"C{i}",
            "oldbalanceOrg": balance,
            "nameDest": f"M{i}",
            "oldbalanceDest": rng.uniform(0, 5_000),
            "isFraud": 0,
        })

    for i in range(n_fraud):
        balance = rng.uniform(100, 10_000)
        rows.append({
            "step": int(rng.integers(1, 500)),
            "type": rng.choice(["TRANSFER", "CASH_OUT"]),
            "amount": balance,  # drains the account -- the real signal
            "nameOrig": f"F{i}",
            "oldbalanceOrg": balance,
            "nameDest": f"D{i}",
            "oldbalanceDest": 0.0,
            "isFraud": 1,
        })

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


@pytest.fixture
def isolated_model_dir(tmp_path, monkeypatch):
    """Points every train.py output path at a scratch directory."""
    monkeypatch.setattr(train_module, "MODEL_PATH", str(tmp_path / "xgb_fraud.json"))
    monkeypatch.setattr(train_module, "SCALER_PATH", str(tmp_path / "scaler.pkl"))
    monkeypatch.setattr(train_module, "FEATURES_PATH", str(tmp_path / "feature_names.pkl"))
    monkeypatch.setattr(train_module, "METRICS_PATH", str(tmp_path / "metrics.json"))
    monkeypatch.setattr(train_module, "MODEL_DIR", str(tmp_path))
    return tmp_path


def test_train_runs_end_to_end_on_small_sample(isolated_model_dir, monkeypatch):
    """
    The full pipeline -- load -> velocity features -> engineer_features ->
    split -> scale -> SMOTE -> fit -> threshold selection -> save -- should
    run without raising on a small, well-formed sample.
    """
    df = _synthetic_paysim_df()
    monkeypatch.setattr(train_module, "load_data", lambda: df)

    train_module.train()

    assert os.path.exists(train_module.MODEL_PATH)
    assert os.path.exists(train_module.SCALER_PATH)
    assert os.path.exists(train_module.FEATURES_PATH)
    assert os.path.exists(os.path.join(isolated_model_dir, "threshold.pkl"))
    assert os.path.exists(os.path.join(isolated_model_dir, "feature_importance.csv"))


def test_train_saves_metrics_file(isolated_model_dir, monkeypatch):
    """
    Regression test for PRD 3.2: train.py used to print ROC-AUC/PR-AUC/etc.
    to stdout only, with no file anyone could read the real numbers back
    from later (which is how the README's stale 0.9997 badge happened).
    """
    df = _synthetic_paysim_df()
    monkeypatch.setattr(train_module, "load_data", lambda: df)

    train_module.train()

    with open(train_module.METRICS_PATH) as f:
        metrics = json.load(f)

    for key in ("roc_auc", "pr_auc", "best_threshold", "n_features",
                "n_train_rows", "n_test_rows", "trained_at_utc"):
        assert key in metrics, f"missing metrics field: {key}"

    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert metrics["n_features"] == len(train_module.FEATURE_COLS)
