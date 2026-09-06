"""
Train XGBoost Fraud Detection Model using the PaySim Dataset
"""

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from imblearn.over_sampling import SMOTE

from models.features import engineer_features, add_velocity_features, FEATURE_COLS

SEED = 42

MODEL_DIR = os.path.dirname(__file__)

DATA_PATH = os.path.join(
    os.path.dirname(MODEL_DIR),
    "data",
    "paysim.csv"
)

MODEL_PATH = os.path.join(MODEL_DIR, "xgb_fraud.json")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
REGISTRY_PATH = os.path.join(MODEL_DIR, "model_registry.jsonl")

# Kaggle's own PaySim page warns that balance columns can carry simulator
# artifacts specifically on fraud rows (fraud transactions get cancelled
# mid-simulation), so even the pre-transaction oldbalanceOrg/oldbalanceDest
# could end up leaking signal, not just the post-transaction ones we already
# excluded. These constants drive a post-training sanity check for that.
LEAKAGE_WATCH_FEATURES = {"log_oldbalanceOrg", "log_oldbalanceDest"}
DOMINANCE_THRESHOLD = 0.40  # one feature owning >=40% of importance is suspicious


def load_data():
    print("Loading PaySim dataset...")

    df = pd.read_csv(
        DATA_PATH,
        usecols=[
            "step",
            "type",
            "amount",
            "nameOrig",
            "oldbalanceOrg",
            "nameDest",
            "oldbalanceDest",
            "isFraud",
        ],
    )
    # newbalanceOrig / newbalanceDest are intentionally NOT loaded -- see
    # the note in models/features.py on why they leak the label in PaySim.

    print(f"Loaded {len(df):,} transactions")
    print(f"Fraud Rate: {df['isFraud'].mean()*100:.4f}%")

    return df

def train():
    df = load_data()

    print("Deriving velocity features (recency_hours, txn_count_24h, is_dest_new)...")
    df = add_velocity_features(df)

    print("Engineering features...")
    df = engineer_features(df)

    X = df[FEATURE_COLS]
    y = df["isFraud"]

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=SEED,
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Applying SMOTE...")
    smote = SMOTE(
        random_state=SEED,
        sampling_strategy=0.10,
    )

    X_train, y_train = smote.fit_resample(X_train, y_train)

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )

    print("Training model...")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    from sklearn.metrics import precision_recall_curve

    probabilities = model.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_test,
        probabilities
    )

    f1_scores = (2 * precision[:-1] * recall[:-1]) / (
        precision[:-1] + recall[:-1] + 1e-8
    )

    best_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_index]

    print(f"\nBest Threshold: {best_threshold:.3f}")

    predictions = (probabilities >= best_threshold).astype(int)

    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)
    report = classification_report(y_test, predictions, output_dict=True)
    cm = confusion_matrix(y_test, predictions)

    print("\n==============================")
    print("Model Evaluation")
    print("==============================")
    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}")

    print("\nClassification Report\n")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix\n")
    print(cm)

    model.save_model(MODEL_PATH)
    
    save_feature_importance(model)
    
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(FEATURE_COLS, FEATURES_PATH)
    
    joblib.dump(best_threshold,
            os.path.join(MODEL_DIR, "threshold.pkl"))

    save_metrics(
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        report=report,
        confusion=cm,
        best_threshold=best_threshold,
        n_train_rows=len(X_train),
        n_test_rows=len(X_test),
    )

    model_version = _compute_model_version(MODEL_PATH)
    append_to_registry(
        model_version=model_version,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        best_threshold=best_threshold,
        n_train_rows=len(X_train),
        n_test_rows=len(X_test),
    )

    print("\nModel saved successfully!")
    print(MODEL_PATH)
    print(f"Metrics saved to {METRICS_PATH}")
    print(f"Model version: {model_version} (see {REGISTRY_PATH})")
    
def save_metrics(roc_auc, pr_auc, report, confusion, best_threshold,
                  n_train_rows, n_test_rows):
    """
    Persists the metrics train.py already prints to stdout, so there's a
    single source of truth for "what's the real accuracy now" instead of
    numbers only living in a terminal scrollback (see PRD 3.2: the README's
    ROC-AUC 0.9997 badge went stale exactly this way after the leakage fix).
    """
    fraud_report = report.get("1", report.get("1.0", {}))

    metrics = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "xgboost_version": xgb.__version__,
        "n_features": len(FEATURE_COLS),
        "n_train_rows": int(n_train_rows),
        "n_test_rows": int(n_test_rows),
        "best_threshold": float(best_threshold),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision_fraud_class": fraud_report.get("precision"),
        "recall_fraud_class": fraud_report.get("recall"),
        "f1_fraud_class": fraud_report.get("f1-score"),
        "accuracy": report.get("accuracy"),
        "confusion_matrix": confusion.tolist(),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)


def _compute_model_version(model_path):
    """
    Content-addressed version id: a short hash of the saved model file
    itself, so the version always reflects exactly which weights are
    deployed -- it can't drift out of sync the way a manually-bumped
    version number could.
    """
    with open(model_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return digest[:12]


def append_to_registry(model_version, roc_auc, pr_auc, best_threshold,
                        n_train_rows, n_test_rows):
    """
    Appends one line per training run to models/model_registry.jsonl --
    an append-only log of every model version ever produced, so past
    versions/metrics aren't lost the moment a new run overwrites
    xgb_fraud.json and metrics.json. Supports PRD §5 "Model versioning /
    registry: track which model version served which prediction, to
    support rollback and A/B comparison as the model evolves."

    This logs training-run history, not live traffic; predict_fraud()
    separately reports model_version on every response so a caller (or
    a future structured-logging layer) can correlate a specific
    prediction back to a row in this file.
    """
    entry = {
        "model_version": model_version,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "best_threshold": float(best_threshold),
        "n_train_rows": int(n_train_rows),
        "n_test_rows": int(n_test_rows),
    }
    with open(REGISTRY_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def save_feature_importance(model):
    importance = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    importance.to_csv(
        os.path.join(MODEL_DIR, "feature_importance.csv"),
        index=False
    )

    print("\nTop Features")
    print(importance.head(10))

    _check_for_leakage(importance)


def _check_for_leakage(importance: pd.DataFrame):
    """
    Heuristic, not a guarantee: flags when a single feature dominates
    importance, since that's a common symptom of label leakage (it's
    exactly how the old newbalanceOrig leak would have looked). Extra
    attention on the balance columns per Kaggle's own note about the
    dataset -- see LEAKAGE_WATCH_FEATURES above.
    """
    total = importance["Importance"].sum()
    if total <= 0:
        return

    top_feature = importance.iloc[0]["Feature"]
    top_share = importance.iloc[0]["Importance"] / total

    print("\n==============================")
    print("Leakage Sanity Check")
    print("==============================")

    if top_share >= DOMINANCE_THRESHOLD:
        note = (
            " (a balance column -- see the leakage note in models/features.py)"
            if top_feature in LEAKAGE_WATCH_FEATURES else ""
        )
        print(
            f"WARNING: '{top_feature}' alone accounts for {top_share:.0%} of "
            f"total feature importance{note}. A single feature dominating "
            f"this much is a common symptom of leakage. Before trusting this "
            f"model, inspect that feature's distribution split by isFraud "
            f"(e.g. df.groupby('isFraud')[<col>].describe())."
        )
    else:
        present = [f for f in LEAKAGE_WATCH_FEATURES if f in importance["Feature"].values]
        watched_share = (
            importance.set_index("Feature").loc[present, "Importance"].sum() / total
            if present else 0.0
        )
        print(
            f"No single feature dominates (top: '{top_feature}' at "
            f"{top_share:.0%}). Balance-column features account for "
            f"{watched_share:.0%} of total importance combined -- within a "
            f"normal range."
        )


if __name__ == "__main__":
    train()