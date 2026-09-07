"""
One-off diagnostic: txn_count_24h and recency_hours show ~0% feature
importance in the trained model (see models/metrics.json ->
feature_importance), despite add_velocity_features() being written
specifically to derive them. This script checks the two most likely
explanations against the real PaySim data:

  1. Low variance / degeneracy: if almost every row has the same value
     (e.g. txn_count_24h == 0 for ~all senders because PaySim senders
     mostly transact once), there's nothing for a tree model to split on.
  2. No relationship with the label: even if the feature does vary,
     it may just not differ between fraud and non-fraud rows in this
     dataset -- printed as a mean/std comparison and point-biserial
     correlation with isFraud.

Usage:
    python scripts/diagnose_velocity_features.py

Requires data/paysim.csv locally (gitignored, not in this repo
checkout -- run wherever the dataset lives).
"""

import os
import sys

import pandas as pd

# See the same fix in scripts/verify_would_drain_orig.py: running this file
# directly puts scripts/ on sys.path, not the repo root, so the models
# import below fails without this.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.features import add_velocity_features, engineer_features

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "paysim.csv")

VELOCITY_COLS = ["recency_hours", "txn_count_24h", "is_dest_new"]


def _describe_variance(df, col):
    print(f"\n--- {col}: distribution ---")
    print(df[col].describe())
    n_unique = df[col].nunique()
    top_value_share = df[col].value_counts(normalize=True).iloc[0]
    print(f"unique values: {n_unique}")
    print(f"most common single value covers {top_value_share:.2%} of rows "
          f"(value = {df[col].value_counts().idxmax()})")
    if top_value_share > 0.95:
        print("^ DEGENERATE: >95% of rows share one value -- a tree model "
              "has almost nothing to split on here, regardless of any true "
              "relationship with fraud.")


def _compare_by_label(df, col):
    print(f"\n--- {col}: split by isFraud ---")
    print(df.groupby("isFraud")[col].agg(["mean", "std", "median"]))
    corr = df[col].corr(df["isFraud"])
    print(f"correlation with isFraud: {corr:.4f}")
    if abs(corr) < 0.02:
        print("^ Essentially no linear relationship with the label in this "
              "dataset -- consistent with the near-zero importance seen "
              "in training.")


def main():
    df = pd.read_csv(
        DATA_PATH,
        usecols=["step", "type", "amount", "nameOrig", "oldbalanceOrg",
                  "nameDest", "oldbalanceDest", "isFraud"],
    )
    print(f"Loaded {len(df):,} rows. Deriving velocity features "
          f"(same as train.py)...")
    df = add_velocity_features(df)
    df = engineer_features(df)

    for col in VELOCITY_COLS:
        print("\n" + "=" * 60)
        print(col)
        print("=" * 60)
        _describe_variance(df, col)
        _compare_by_label(df, col)

    print(
        "\n" + "=" * 60 +
        "\nInterpretation" +
        "\n" + "=" * 60 +
        "\nIf a feature is degenerate (one value dominates >95% of rows), "
        "that alone explains the near-zero importance -- likely because "
        "PaySim simulates most senders transacting only once or twice in "
        "the whole 744-step run, so txn_count_24h == 0 and "
        "recency_hours == 720 (the 'no prior transaction' default) for "
        "almost everyone. That's a property of this synthetic dataset, not "
        "necessarily of real transaction velocity as a fraud signal -- in "
        "production, with a real transaction log, these fields would likely "
        "have far more spread. Worth noting in the README as a known "
        "limitation of training on PaySim rather than dropping the features "
        "outright, since a live deployment would supply real values for "
        "them."
    )


if __name__ == "__main__":
    main()
