"""
One-off verification script for PRD §3.1: `would_drain_orig` accounts for
57% of feature importance in the retrained model -- above the 40%
dominance threshold train.py's own leakage check warns on. This script
runs the actual recommended check (distribution split by isFraud) against
the real PaySim data so that number can be trusted or flagged.

Usage:
    python scripts/verify_would_drain_orig.py

Requires data/paysim.csv to be present locally (it's gitignored and not
in this repo checkout, so this couldn't be run from within the chat
session that wrote this script -- run it wherever the dataset lives).
"""

import os

import pandas as pd

from models.features import engineer_features

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "paysim.csv")


def main():
    df = pd.read_csv(
        DATA_PATH,
        usecols=["type", "amount", "oldbalanceOrg", "oldbalanceDest", "isFraud"],
    )
    df = engineer_features(df)

    print("=" * 60)
    print("would_drain_orig split by isFraud")
    print("=" * 60)
    print(df.groupby("isFraud")["would_drain_orig"].describe())

    # Cross-tab is the sharpest version of the same question: of the rows
    # where would_drain_orig == 1, what fraction are actually fraud, and
    # vice versa? If it were leaking the label the way the old
    # newbalanceOrig feature did, we'd expect near-perfect separation
    # (something close to 100% of fraud rows have would_drain_orig == 1
    # AND something close to 100% of would_drain_orig == 1 rows are fraud).
    # A strong-but-imperfect signal (elevated fraud rate among
    # would_drain_orig==1, but plenty of legitimate full-balance transfers
    # too) is what a genuine prospective signal looks like instead.
    print("\n" + "=" * 60)
    print("Cross-tab: would_drain_orig vs isFraud (row counts)")
    print("=" * 60)
    print(pd.crosstab(df["would_drain_orig"], df["isFraud"]))

    print("\n" + "=" * 60)
    print("Cross-tab: would_drain_orig vs isFraud (row %, normalized by would_drain_orig)")
    print("=" * 60)
    print(pd.crosstab(df["would_drain_orig"], df["isFraud"], normalize="index"))

    fraud_rate_overall = df["isFraud"].mean()
    fraud_rate_when_drained = df.loc[df["would_drain_orig"] == 1, "isFraud"].mean()
    fraud_rate_when_not_drained = df.loc[df["would_drain_orig"] == 0, "isFraud"].mean()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Overall fraud rate:                    {fraud_rate_overall:.4%}")
    print(f"Fraud rate when would_drain_orig == 1:  {fraud_rate_when_drained:.4%}")
    print(f"Fraud rate when would_drain_orig == 0:  {fraud_rate_when_not_drained:.4%}")
    print(
        "\nIf fraud rate at would_drain_orig==1 is elevated but well short of "
        "100%, and a meaningful share of isFraud==1 rows have "
        "would_drain_orig==0 (fraud attempts that DON'T drain the account -- "
        "check the crosstab above), that's consistent with a real prospective "
        "signal rather than a leak. If instead nearly every fraud row has "
        "would_drain_orig==1 AND nearly every would_drain_orig==1 row is "
        "fraud, treat that the same way the old newbalanceOrig leak was "
        "treated and dig further before trusting the model."
    )


if __name__ == "__main__":
    main()
