"""
train.py
--------
Generates synthetic financial transaction data (6.3 M+ rows),
engineers features, trains XGBoost fraud detection model,
and saves the model + scaler to disk.

Run:  python models/train.py
"""

import os
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

SEED = 42
np.random.seed(SEED)
N_SAMPLES = 6_300_000          # 6.3 M transactions
FRAUD_RATE = 0.002             # 0.2 % fraud (realistic)

MODEL_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_fraud.json")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")


# ─────────────────────────────────────────────
# 1.  Generate synthetic data
# ─────────────────────────────────────────────
def generate_data(n: int = N_SAMPLES) -> pd.DataFrame:
    print(f"[DATA] Generating {n:,} synthetic transactions …")
    t0 = time.time()

    txn_types = np.random.choice(
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"],
        size=n, p=[0.35, 0.20, 0.20, 0.15, 0.10]
    )

    old_balance_orig = np.random.exponential(scale=15_000, size=n).clip(0)
    amount = np.random.exponential(scale=8_000, size=n).clip(1)
    new_balance_orig = (old_balance_orig - amount).clip(0)

    old_balance_dest = np.random.exponential(scale=20_000, size=n).clip(0)
    new_balance_dest = old_balance_dest + amount

    # Fraud label — TRANSFER / CASH_OUT are higher risk
    base_fraud_prob = np.where(
        np.isin(txn_types, ["TRANSFER", "CASH_OUT"]), 0.004, 0.0005
    )
    is_fraud = np.random.binomial(1, base_fraud_prob)

    # Recency & velocity features
    recency_hours = np.random.exponential(scale=24, size=n).clip(0, 720)
    txn_count_24h = np.random.poisson(lam=3, size=n).clip(0, 50)
    is_dest_new   = np.random.binomial(1, 0.15, size=n)

    df = pd.DataFrame({
        "type":              txn_types,
        "amount":            amount,
        "oldbalanceOrg":     old_balance_orig,
        "newbalanceOrig":    new_balance_orig,
        "oldbalanceDest":    old_balance_dest,
        "newbalanceDest":    new_balance_dest,
        "recency_hours":     recency_hours,
        "txn_count_24h":     txn_count_24h,
        "is_dest_new":       is_dest_new,
        "isFraud":           is_fraud,
    })

    print(f"[DATA] Done in {time.time()-t0:.1f}s  |  Fraud rate: {is_fraud.mean()*100:.3f}%")
    return df


# ─────────────────────────────────────────────
# 2.  Feature engineering
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Encode transaction type
    type_map = {"PAYMENT": 0, "TRANSFER": 1, "CASH_OUT": 2, "DEBIT": 3, "CASH_IN": 4}
    df["type_enc"] = df["type"].map(type_map).fillna(0).astype(int)

    # Balance deltas
    df["balance_change_orig"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
    df["balance_change_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    # Account drained flag (balance goes to ~0)
    df["orig_drained"] = (
        (df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)
    ).astype(int)

    # Amount relative to sender balance
    df["amount_ratio_orig"] = np.where(
        df["oldbalanceOrg"] > 0,
        df["amount"] / (df["oldbalanceOrg"] + 1),
        0
    )

    # Destination balance anomaly
    df["dest_balance_anomaly"] = np.where(
        df["oldbalanceDest"] == 0, 1, 0
    )

    # Log-transform skewed columns
    for col in ["amount", "oldbalanceOrg", "newbalanceOrig",
                "oldbalanceDest", "newbalanceDest"]:
        df[f"log_{col}"] = np.log1p(df[col])

    return df


FEATURE_COLS = [
    "type_enc", "log_amount", "log_oldbalanceOrg", "log_newbalanceOrig",
    "log_oldbalanceDest", "log_newbalanceDest",
    "balance_change_orig", "balance_change_dest",
    "orig_drained", "amount_ratio_orig", "dest_balance_anomaly",
    "recency_hours", "txn_count_24h", "is_dest_new",
]


# ─────────────────────────────────────────────
# 3.  Train
# ─────────────────────────────────────────────
def train():
    df = generate_data()
    df = engineer_features(df)

    X = df[FEATURE_COLS]
    y = df["isFraud"]

    print("[SPLIT] Train / test split …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # SMOTE on a 200k subsample to balance classes
    print("[SMOTE] Balancing classes …")
    sm = SMOTE(random_state=SEED, sampling_strategy=0.1)
    X_res, y_res = sm.fit_resample(X_train_s[:200_000], y_train[:200_000])

    # XGBoost
    print("[TRAIN] Fitting XGBoost …")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=len(y_train[y_train==0]) / max(len(y_train[y_train==1]), 1),
        use_label_encoder=False,
        eval_metric="auc",
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(
        X_res, y_res,
        eval_set=[(X_test_s, y_test)],
        verbose=50,
    )

    # Evaluate
    y_pred_proba = model.predict_proba(X_test_s)[:, 1]
    y_pred       = (y_pred_proba >= 0.5).astype(int)

    auc       = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)

    print("\n" + "="*50)
    print(f"  AUC       : {auc:.4f}")
    print(f"  Precision : {precision*100:.1f}%")
    print(f"  Recall    : {recall*100:.1f}%")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    # Save artefacts
    model.save_model(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(FEATURE_COLS, FEATURES_PATH)
    print(f"[SAVE] Model → {MODEL_PATH}")
    print(f"[SAVE] Scaler → {SCALER_PATH}")


if __name__ == "__main__":
    train()
