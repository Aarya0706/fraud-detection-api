import numpy as np
import pandas as pd

TYPE_MAP = {
    "PAYMENT": 0,
    "TRANSFER": 1,
    "CASH_OUT": 2,
    "DEBIT": 3,
    "CASH_IN": 4,
}

# NOTE: newbalanceOrig / newbalanceDest are deliberately NOT used. Those are
# POST-transaction balances. In PaySim they leak the label almost perfectly
# (fraudulent TRANSFER/CASH_OUT transactions drain the origin account to
# exactly 0), which is why the old model hit ~0.9999 AUC -- it was mostly
# learning "did this account get drained" rather than real fraud patterns.
# In production you also don't have the post-transaction balance yet when
# deciding whether to allow the transaction. Every feature below must be
# computable BEFORE the transaction is approved.

FEATURE_COLS = [
    "type_enc",
    "log_amount",
    "log_oldbalanceOrg",
    "log_oldbalanceDest",
    "amount_ratio_orig",
    "would_drain_orig",
    "dest_balance_anomaly",
    "is_dest_new",
    "recency_hours",
    "txn_count_24h",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turns raw pre-transaction fields into model features.

    Expects df to already contain: type, amount, oldbalanceOrg,
    oldbalanceDest, recency_hours, txn_count_24h, is_dest_new.
    The last three are velocity/history features. At inference time the
    caller supplies them directly (a live system tracks its own transaction
    log). At training time, `add_velocity_features` below derives them from
    the PaySim simulation log.
    """
    df = df.copy()

    df["type_enc"] = df["type"].map(TYPE_MAP)
    if df["type_enc"].isna().any():
        bad = sorted(set(df.loc[df["type_enc"].isna(), "type"]))
        raise ValueError(f"Unknown transaction type(s): {bad}")
    df["type_enc"] = df["type_enc"].astype(int)

    # Prospective drain: would this transaction, if approved, leave the
    # sender at or below zero? Uses only oldbalanceOrg + the requested
    # amount -- both known before approval -- instead of the leaky actual
    # post-transaction balance.
    df["would_drain_orig"] = (
        (df["oldbalanceOrg"] > 0) &
        ((df["oldbalanceOrg"] - df["amount"]) <= 0)
    ).astype(int)

    df["amount_ratio_orig"] = np.where(
        df["oldbalanceOrg"] > 0,
        df["amount"] / (df["oldbalanceOrg"] + 1),
        0,
    )

    df["dest_balance_anomaly"] = (
        df["oldbalanceDest"] == 0
    ).astype(int)

    for col in ["amount", "oldbalanceOrg", "oldbalanceDest"]:
        df[f"log_{col}"] = np.log1p(df[col])

    return df


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    TRAINING-ONLY helper. Derives recency_hours / txn_count_24h / is_dest_new
    from PaySim's raw simulation columns (step, nameOrig, nameDest) so the
    training data has the same fields a live caller would supply.

    PaySim's `step` = 1 simulated hour, so step deltas are already hours.
    """
    df = df.copy()

    # --- sender-side velocity: needs each sender's own rows in time order ---
    by_sender = df.sort_values(["nameOrig", "step"])
    prev_step = by_sender.groupby("nameOrig")["step"].shift(1)
    # No prior transaction -> treat as a long-dormant/new sender (720h = 30d).
    recency = (by_sender["step"] - prev_step).fillna(720).clip(lower=0)

    # txn_count_24h: count of this sender's OTHER transactions in the
    # trailing 24h, excluding the current one.
    #
    # PaySim has ~6.3M rows and most senders only transact once or twice, so
    # a per-group groupby().apply(python_function) means ~6M Python-level
    # function calls -- far too slow (this is what was hanging earlier).
    # Instead, compute it in one vectorized pass over the whole array: give
    # each group a numeric id and fold it into the step value with a large
    # offset, so groups never overlap in the combined key. A single global
    # np.searchsorted then behaves exactly like a per-group one, with zero
    # Python-level looping over groups.
    steps = by_sender["step"].to_numpy(dtype=np.int64)
    names = by_sender["nameOrig"].to_numpy()

    new_group = np.empty(len(names), dtype=bool)
    new_group[0] = True
    new_group[1:] = names[1:] != names[:-1]
    group_id = np.cumsum(new_group) - 1

    OFFSET = 100_000  # >> max possible step range (PaySim runs 0..744)
    combined = group_id * OFFSET + steps
    combined_low = group_id * OFFSET + (steps - 24)

    counts = (
        np.searchsorted(combined, combined, side="left") -
        np.searchsorted(combined, combined_low, side="left")
    )
    txn_count = pd.Series(counts, index=by_sender.index)

    df["recency_hours"] = recency.reindex(df.index)
    df["txn_count_24h"] = txn_count.reindex(df.index)

    # --- destination-side novelty: needs GLOBAL time order, not per-sender
    # order, or a dest could be mislabeled "new" just because of how the
    # per-sender sort happened to interleave rows.
    by_time = df.sort_values("step")
    is_new = ~by_time["nameDest"].duplicated()
    df["is_dest_new"] = is_new.reindex(df.index).astype(int)

    return df