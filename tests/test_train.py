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
        # Spread fraud rows evenly across the full step range (rather than
        # drawing them uniformly at random, which can by chance leave one
        # side of a chronological split with zero fraud rows in a sample
        # this small) so both the time-aware and random split strategies
        # have fraud examples on both sides of the split in tests.
        step = int(1 + (i / max(n_fraud - 1, 1)) * 490)
        rows.append({
            "step": step,
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
    monkeypatch.setattr(train_module, "REGISTRY_PATH", str(tmp_path / "model_registry.jsonl"))
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


def test_train_appends_to_model_registry(isolated_model_dir, monkeypatch):
    """
    PRD §5 "Model versioning / registry": every training run should add
    one line to model_registry.jsonl (not overwrite it), so past model
    versions and their metrics survive later retrains -- unlike
    xgb_fraud.json / metrics.json, which get overwritten each run.
    """
    df = _synthetic_paysim_df()
    monkeypatch.setattr(train_module, "load_data", lambda: df)

    train_module.train()
    train_module.train()  # second run -- registry should now have 2 lines

    with open(train_module.REGISTRY_PATH) as f:
        lines = [line for line in f if line.strip()]

    assert len(lines) == 2
    entries = [json.loads(line) for line in lines]
    for entry in entries:
        for key in ("model_version", "trained_at_utc", "roc_auc",
                    "pr_auc", "best_threshold", "n_train_rows", "n_test_rows"):
            assert key in entry

    # Same model file re-hashed identically-trained data with the same
    # SEED -> versions may legitimately collide, but the field itself
    # must always be a non-empty content hash, never blank/placeholder.
    assert all(len(e["model_version"]) > 0 for e in entries)
    


# ── Threshold calibration by business cost (PRD §5) ────────────────

def test_select_threshold_by_cost_prefers_higher_recall_when_fn_costlier():
    """
    With false negatives (missed fraud) priced far above false positives,
    the cost-minimizing threshold should sit at or below the max-F1
    threshold on this obviously-separable toy example -- i.e. it should
    lean towards catching more fraud rather than fewer false alarms.
    """
    y_true = np.array([0] * 8 + [1] * 2)
    # Fraud rows (index 8, 9) score higher; a couple of legit rows score
    # in the middle to make the threshold choice actually matter.
    probabilities = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.4, 0.4, 0.9, 0.95])

    threshold, cost, curve = train_module._select_threshold_by_cost(
        y_true, probabilities, cost_fp=1, cost_fn=100
    )

    assert 0.0 < threshold <= 0.9
    assert cost >= 0
    assert len(curve) > 0
    assert all({"threshold", "cost", "fp", "fn"} <= c.keys() for c in curve)


def test_select_threshold_by_cost_prefers_higher_precision_when_fp_costlier():
    """Flip the cost ratio: expensive false positives should push the
    selected threshold higher than when false negatives dominate."""
    y_true = np.array([0] * 8 + [1] * 2)
    probabilities = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.4, 0.4, 0.9, 0.95])

    fn_expensive_threshold, _, _ = train_module._select_threshold_by_cost(
        y_true, probabilities, cost_fp=1, cost_fn=100
    )
    fp_expensive_threshold, _, _ = train_module._select_threshold_by_cost(
        y_true, probabilities, cost_fp=100, cost_fn=1
    )

    assert fp_expensive_threshold >= fn_expensive_threshold


def test_train_records_both_thresholds_and_strategy(isolated_model_dir, monkeypatch):
    """
    Regression test for PRD §5 "Threshold calibration by business cost":
    metrics.json and the registry should record the max-F1 threshold, the
    cost-based threshold, and which strategy was actually selected --
    not just a single unlabeled number, so the tradeoff stays visible.
    """
    df = _synthetic_paysim_df()
    monkeypatch.setattr(train_module, "load_data", lambda: df)
    monkeypatch.setattr(train_module, "THRESHOLD_STRATEGY", "f1")

    train_module.train()

    with open(train_module.METRICS_PATH) as f:
        metrics = json.load(f)

    assert metrics["threshold_strategy"] == "f1"
    assert metrics["best_threshold"] == pytest.approx(metrics["f1_threshold"])
    assert "cost_threshold" in metrics
    assert "cost_threshold_assumptions" in metrics
    for key in ("cost_false_positive", "cost_false_negative", "total_cost_at_cost_threshold"):
        assert key in metrics["cost_threshold_assumptions"]

    with open(train_module.REGISTRY_PATH) as f:
        entry = json.loads(f.readline())
    assert entry["threshold_strategy"] == "f1"
    assert "f1_threshold" in entry and "cost_threshold" in entry


def test_train_cost_strategy_selects_cost_threshold(isolated_model_dir, monkeypatch):
    """When THRESHOLD_STRATEGY='cost', the saved best_threshold/threshold.pkl
    should track the cost-based threshold, not the max-F1 one."""
    df = _synthetic_paysim_df()
    monkeypatch.setattr(train_module, "load_data", lambda: df)
    monkeypatch.setattr(train_module, "THRESHOLD_STRATEGY", "cost")

    train_module.train()

    with open(train_module.METRICS_PATH) as f:
        metrics = json.load(f)

    assert metrics["threshold_strategy"] == "cost"
    assert metrics["best_threshold"] == pytest.approx(metrics["cost_threshold"])


# ── Time-aware evaluation split ─────────────────────────────────────

def test_time_split_is_chronologically_non_overlapping():
    """
    recency_hours/txn_count_24h are derived from the full event log's time
    order, so the default split must hold out the *latest* slice of steps
    as test -- every training step must come no later than every test
    step, unlike a random/stratified split which interleaves them.
    """
    df = _synthetic_paysim_df()
    from models.features import engineer_features, add_velocity_features
    df = add_velocity_features(df)
    df = engineer_features(df)

    X_train, X_test, y_train, y_test = train_module._split_dataset(df)

    train_idx = X_train.index
    test_idx = X_test.index
    assert df.loc[train_idx, "step"].max() <= df.loc[test_idx, "step"].min()
    assert y_train.sum() > 0 and y_test.sum() > 0
    assert len(X_train) + len(X_test) == len(df)


def test_random_split_strategy_still_available(monkeypatch):
    """SPLIT_STRATEGY=random preserves the original stratified IID split,
    for comparison / back-compat with prior training runs."""
    df = _synthetic_paysim_df()
    from models.features import engineer_features, add_velocity_features
    df = add_velocity_features(df)
    df = engineer_features(df)

    monkeypatch.setattr(train_module, "SPLIT_STRATEGY", "random")
    X_train, X_test, y_train, y_test = train_module._split_dataset(df)

    assert len(X_train) + len(X_test) == len(df)
    assert y_train.sum() > 0 and y_test.sum() > 0


def test_train_records_split_strategy(isolated_model_dir, monkeypatch):
    """metrics.json and the registry should record which split strategy
    produced the reported numbers, since it changes what the ROC-AUC/PR-AUC
    figures actually mean for this kind of time-derived-feature model."""
    df = _synthetic_paysim_df()
    monkeypatch.setattr(train_module, "load_data", lambda: df)

    train_module.train()

    with open(train_module.METRICS_PATH) as f:
        metrics = json.load(f)
    assert metrics["split_strategy"] == "time"

    with open(train_module.REGISTRY_PATH) as f:
        entry = json.loads(f.readline())
    assert entry["split_strategy"] == "time"