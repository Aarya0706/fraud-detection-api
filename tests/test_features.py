"""
Unit tests for models/features.py

Covers:
  - engineer_features: correctness of each derived column, and that
    unknown `type` values are rejected rather than silently defaulted.
  - add_velocity_features: recency_hours / txn_count_24h / is_dest_new
    derivation from a small hand-built PaySim-shaped log.
"""

import numpy as np
import pandas as pd
import pytest

from models.features import engineer_features, add_velocity_features, FEATURE_COLS


# ── engineer_features ──────────────────────────────────────────────

def _base_row(**overrides):
    row = {
        "type": "PAYMENT",
        "amount": 100.0,
        "oldbalanceOrg": 500.0,
        "oldbalanceDest": 0.0,
        "recency_hours": 24.0,
        "txn_count_24h": 1,
        "is_dest_new": 0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_feature_cols_are_all_produced():
    df = engineer_features(_base_row())
    for col in FEATURE_COLS:
        assert col in df.columns, f"missing engineered column: {col}"


def test_no_leaky_post_transaction_columns():
    # newbalanceOrig / newbalanceDest must never be required inputs or
    # outputs of engineer_features -- they are post-transaction and leak
    # the label in PaySim (see the leakage note in features.py).
    df = engineer_features(_base_row())
    assert "newbalanceOrig" not in df.columns
    assert "newbalanceDest" not in df.columns
    assert "log_newbalanceOrig" not in df.columns
    assert "log_newbalanceDest" not in df.columns


@pytest.mark.parametrize("type_str,expected_code", [
    ("PAYMENT", 0),
    ("TRANSFER", 1),
    ("CASH_OUT", 2),
    ("DEBIT", 3),
    ("CASH_IN", 4),
])
def test_type_encoding(type_str, expected_code):
    df = engineer_features(_base_row(type=type_str))
    assert df["type_enc"].iloc[0] == expected_code


def test_unknown_type_is_rejected_not_defaulted():
    """
    Regression test for the fixed bug: unknown/typo'd `type` values used to
    silently map to PAYMENT (code 0, the lowest-risk class) via .fillna(0).
    Now they must raise instead.
    """
    with pytest.raises(ValueError, match="Unknown transaction type"):
        engineer_features(_base_row(type="WIRE_TRANSFER"))


def test_would_drain_orig_true_when_amount_meets_or_exceeds_balance():
    df = engineer_features(_base_row(oldbalanceOrg=100.0, amount=100.0))
    assert df["would_drain_orig"].iloc[0] == 1

    df = engineer_features(_base_row(oldbalanceOrg=100.0, amount=150.0))
    assert df["would_drain_orig"].iloc[0] == 1


def test_would_drain_orig_false_when_balance_remains():
    df = engineer_features(_base_row(oldbalanceOrg=1000.0, amount=100.0))
    assert df["would_drain_orig"].iloc[0] == 0


def test_would_drain_orig_false_when_sender_already_at_zero():
    # oldbalanceOrg == 0 means there was nothing to drain in the first place.
    df = engineer_features(_base_row(oldbalanceOrg=0.0, amount=0.01))
    assert df["would_drain_orig"].iloc[0] == 0


def test_dest_balance_anomaly_flags_zero_balance_destination():
    df = engineer_features(_base_row(oldbalanceDest=0.0))
    assert df["dest_balance_anomaly"].iloc[0] == 1

    df = engineer_features(_base_row(oldbalanceDest=250.0))
    assert df["dest_balance_anomaly"].iloc[0] == 0


def test_amount_ratio_orig_computation():
    df = engineer_features(_base_row(oldbalanceOrg=100.0, amount=50.0))
    expected = 50.0 / (100.0 + 1)
    assert df["amount_ratio_orig"].iloc[0] == pytest.approx(expected)


def test_amount_ratio_orig_zero_when_sender_balance_zero():
    df = engineer_features(_base_row(oldbalanceOrg=0.0, amount=50.0))
    assert df["amount_ratio_orig"].iloc[0] == 0


def test_log_transforms_are_log1p():
    df = engineer_features(_base_row(amount=999.0, oldbalanceOrg=500.0, oldbalanceDest=0.0))
    assert df["log_amount"].iloc[0] == pytest.approx(np.log1p(999.0))
    assert df["log_oldbalanceOrg"].iloc[0] == pytest.approx(np.log1p(500.0))
    assert df["log_oldbalanceDest"].iloc[0] == pytest.approx(np.log1p(0.0))


# ── add_velocity_features ──────────────────────────────────────────

def test_velocity_features_first_txn_is_dormant_and_dest_is_new():
    df = pd.DataFrame([
        {"step": 10, "nameOrig": "A1", "nameDest": "B1", "amount": 100, "type": "PAYMENT"},
    ])
    out = add_velocity_features(df)
    assert out["recency_hours"].iloc[0] == 720  # no prior txn -> dormant default
    assert out["txn_count_24h"].iloc[0] == 0    # no OTHER txns yet to count
    assert out["is_dest_new"].iloc[0] == 1


def test_velocity_features_recency_and_count_for_repeat_sender():
    df = pd.DataFrame([
        {"step": 0,  "nameOrig": "A1", "nameDest": "B1", "amount": 100, "type": "PAYMENT"},
        {"step": 5,  "nameOrig": "A1", "nameDest": "B2", "amount": 200, "type": "PAYMENT"},
        {"step": 30, "nameOrig": "A1", "nameDest": "B3", "amount": 300, "type": "PAYMENT"},
    ])
    out = add_velocity_features(df).sort_values("step").reset_index(drop=True)

    # second txn: 5h since the first
    assert out["recency_hours"].iloc[1] == 5
    # third txn: 25h since the second (30 - 5), outside the 24h window of txn 2
    assert out["recency_hours"].iloc[2] == 25

    # txn_count_24h excludes the current transaction (per docstring: "OTHER
    # transactions ... excluding the current one").
    assert out["txn_count_24h"].iloc[0] == 0   # step 0: no prior txns at all
    assert out["txn_count_24h"].iloc[1] == 1   # step 5: step 0 is within 24h
    assert out["txn_count_24h"].iloc[2] == 0   # step 30: step 5 is 25h back, outside window


def test_velocity_features_repeat_destination_is_not_new():
    df = pd.DataFrame([
        {"step": 0, "nameOrig": "A1", "nameDest": "B1", "amount": 100, "type": "PAYMENT"},
        {"step": 1, "nameOrig": "A2", "nameDest": "B1", "amount": 100, "type": "PAYMENT"},
    ])
    out = add_velocity_features(df).sort_values("step").reset_index(drop=True)
    assert out["is_dest_new"].iloc[0] == 1  # first time B1 is seen
    assert out["is_dest_new"].iloc[1] == 0  # B1 seen again
