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
    roc_curve,
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

# ── Threshold calibration by business cost (PRD §5) ─────────────────
# Max-F1 picks a threshold that's statistically balanced, not one that
# reflects what a false positive vs. a false negative actually *costs* a
# real deployment. These defaults are illustrative placeholders -- a
# real deployment would set them from actual $ figures (e.g. average
# cost of a manual fraud review vs. average confirmed-fraud loss
# amount). Override without a code change via env vars.
COST_FALSE_POSITIVE = float(os.environ.get("COST_FALSE_POSITIVE", 5))
COST_FALSE_NEGATIVE = float(os.environ.get("COST_FALSE_NEGATIVE", 100))
# Which threshold actually gets saved to threshold.pkl and used live.
# "f1" (default) preserves existing behavior; "cost" switches to the
# cost-minimizing threshold below. Mirrors the API_KEY pattern elsewhere
# in this project: the capability exists, but rollout is an explicit
# opt-in decision, not a silent behavior change.
THRESHOLD_STRATEGY = os.environ.get("THRESHOLD_STRATEGY", "f1").lower()


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

def _select_threshold_by_cost(y_true, probabilities, cost_fp, cost_fn,
                               candidate_thresholds=None):
    """
    Picks the threshold that minimizes total business cost
    (false_positives * cost_fp + false_negatives * cost_fn), instead of
    the max-F1 threshold's purely statistical optimum. Fraud false
    negatives (a missed fraud) are typically far costlier than false
    positives (a legitimate transaction flagged for review), so
    COST_FALSE_NEGATIVE defaults higher than COST_FALSE_POSITIVE --
    tune both to whatever a real deployment's actual costs are.

    Returns (best_threshold, best_cost, full_cost_curve) so the caller
    can both use the selected threshold and inspect/plot the tradeoff.
    """
    if candidate_thresholds is None:
        candidate_thresholds = np.linspace(0.01, 0.99, 99)

    y_true = np.asarray(y_true)
    best_threshold = float(candidate_thresholds[0])
    best_cost = float("inf")
    cost_curve = []

    for t in candidate_thresholds:
        preds = (probabilities >= t).astype(int)
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))
        cost = fp * cost_fp + fn * cost_fn
        cost_curve.append({"threshold": float(t), "cost": float(cost), "fp": fp, "fn": fn})
        if cost < best_cost:
            best_cost = float(cost)
            best_threshold = float(t)

    return best_threshold, best_cost, cost_curve


def _thin_curve(*arrays, max_points=60):
    """
    Downsamples one or more equal-length curve arrays to at most
    max_points, keeping the first and last point, so metrics.json stays
    small enough to serve over /model/info without needing a dedicated
    endpoint or pagination. Purely a display-resolution reduction --
    doesn't change AUC, which is computed separately on the full curve.
    """
    n = len(arrays[0])
    if n <= max_points:
        idx = np.arange(n)
    else:
        idx = np.unique(np.linspace(0, n - 1, max_points).astype(int))
    return [np.asarray(a)[idx].tolist() for a in arrays]


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

    fpr, tpr, _roc_thresholds = roc_curve(y_test, probabilities)

    f1_scores = (2 * precision[:-1] * recall[:-1]) / (
        precision[:-1] + recall[:-1] + 1e-8
    )

    best_index = np.argmax(f1_scores)
    f1_threshold = float(thresholds[best_index])

    cost_threshold, cost_value, _cost_curve = _select_threshold_by_cost(
        y_test, probabilities, COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE
    )

    selected_threshold = f1_threshold if THRESHOLD_STRATEGY != "cost" else cost_threshold

    print("\nThreshold comparison:")
    print(f"  Max-F1 threshold   : {f1_threshold:.3f}")
    print(
        f"  Cost-based threshold: {cost_threshold:.3f}  "
        f"(assumes cost_fp={COST_FALSE_POSITIVE}, cost_fn={COST_FALSE_NEGATIVE}, "
        f"total cost={cost_value:.0f})"
    )
    print(f"  Selected strategy  : '{THRESHOLD_STRATEGY}' -> using {selected_threshold:.3f}")

    best_threshold = selected_threshold
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

    thinned_precision, thinned_recall = _thin_curve(precision[:-1], recall[:-1])
    thinned_fpr, thinned_tpr = _thin_curve(fpr, tpr)

    feature_importance_list = sorted(
        (
            {"feature": f, "importance": float(imp)}
            for f, imp in zip(FEATURE_COLS, model.feature_importances_)
        ),
        key=lambda row: row["importance"],
        reverse=True,
    )

    save_metrics(
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        report=report,
        confusion=cm,
        best_threshold=best_threshold,
        f1_threshold=f1_threshold,
        cost_threshold=cost_threshold,
        cost_value=cost_value,
        threshold_strategy=THRESHOLD_STRATEGY,
        n_train_rows=len(X_train),
        n_test_rows=len(X_test),
        roc_curve_points={"fpr": thinned_fpr, "tpr": thinned_tpr},
        pr_curve_points={"precision": thinned_precision, "recall": thinned_recall},
        feature_importance=feature_importance_list,
    )

    model_version = _compute_model_version(MODEL_PATH)
    append_to_registry(
        model_version=model_version,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        best_threshold=best_threshold,
        f1_threshold=f1_threshold,
        cost_threshold=cost_threshold,
        threshold_strategy=THRESHOLD_STRATEGY,
        n_train_rows=len(X_train),
        n_test_rows=len(X_test),
    )

    print("\nModel saved successfully!")
    print(MODEL_PATH)
    print(f"Metrics saved to {METRICS_PATH}")
    print(f"Model version: {model_version} (see {REGISTRY_PATH})")
    
def save_metrics(roc_auc, pr_auc, report, confusion, best_threshold,
                  f1_threshold, cost_threshold, cost_value, threshold_strategy,
                  n_train_rows, n_test_rows, roc_curve_points=None,
                  pr_curve_points=None, feature_importance=None):
    """
    Persists the metrics train.py already prints to stdout, so there's a
    single source of truth for "what's the real accuracy now" instead of
    numbers only living in a terminal scrollback (see PRD 3.2: the README's
    ROC-AUC 0.9997 badge went stale exactly this way after the leakage fix).

    Records both the max-F1 and cost-based thresholds (and which one was
    actually selected) so a later reviewer can see the tradeoff instead of
    just the winner -- see PRD §5 "Threshold calibration by business cost".

    roc_curve_points / pr_curve_points / feature_importance are optional
    so older callers (and any code depending on this signature) keep
    working -- added purely so the Insights tab can render real analytics
    from this same file via GET /model/info, with no new endpoint.
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
        "threshold_strategy": threshold_strategy,
        "f1_threshold": float(f1_threshold),
        "cost_threshold": float(cost_threshold),
        "cost_threshold_assumptions": {
            "cost_false_positive": COST_FALSE_POSITIVE,
            "cost_false_negative": COST_FALSE_NEGATIVE,
            "total_cost_at_cost_threshold": float(cost_value),
        },
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision_fraud_class": fraud_report.get("precision"),
        "recall_fraud_class": fraud_report.get("recall"),
        "f1_fraud_class": fraud_report.get("f1-score"),
        "accuracy": report.get("accuracy"),
        "confusion_matrix": confusion.tolist(),
        "roc_curve": roc_curve_points,
        "pr_curve": pr_curve_points,
        "feature_importance": feature_importance,
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
                        f1_threshold, cost_threshold, threshold_strategy,
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
        "threshold_strategy": threshold_strategy,
        "f1_threshold": float(f1_threshold),
        "cost_threshold": float(cost_threshold),
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