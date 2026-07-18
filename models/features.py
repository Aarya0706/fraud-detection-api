import numpy as np
import pandas as pd

TYPE_MAP = {
    "PAYMENT": 0,
    "TRANSFER": 1,
    "CASH_OUT": 2,
    "DEBIT": 3,
    "CASH_IN": 4,
}


FEATURE_COLS = [
    "type_enc",
    "log_amount",
    "log_oldbalanceOrg",
    "log_newbalanceOrig",
    "log_oldbalanceDest",
    "log_newbalanceDest",
    "balance_change_orig",
    "balance_change_dest",
    "orig_drained",
    "amount_ratio_orig",
    "dest_balance_anomaly",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["type_enc"] = (
        df["type"]
        .map(TYPE_MAP)
        .fillna(0)
        .astype(int)
    )

    df["balance_change_orig"] = (
        df["newbalanceOrig"] -
        df["oldbalanceOrg"]
    )

    df["balance_change_dest"] = (
        df["newbalanceDest"] -
        df["oldbalanceDest"]
    )

    df["orig_drained"] = (
        (df["oldbalanceOrg"] > 0) &
        (df["newbalanceOrig"] == 0)
    ).astype(int)

    df["amount_ratio_orig"] = np.where(
        df["oldbalanceOrg"] > 0,
        df["amount"] / (df["oldbalanceOrg"] + 1),
        0,
    )

    df["dest_balance_anomaly"] = (
        df["oldbalanceDest"] == 0
    ).astype(int)

    for col in [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]:
        df[f"log_{col}"] = np.log1p(df[col])

    return df