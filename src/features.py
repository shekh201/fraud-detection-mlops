import numpy as np
import pandas as pd


def create_features(df):
    """
    Create fraud detection features from raw transaction data.
    """

    df = df.copy()

    # 1. Sender balance change
    df["balance_change"] = (
        df["oldbalanceOrg"] - df["newbalanceOrig"]
    )

    # 2. Balance consistency check
    df["balance_error"] = (
        df["oldbalanceOrg"]
        - df["amount"]
        - df["newbalanceOrig"]
    )

    # 3. Transaction amount relative to sender balance
    df["amount_balance_ratio"] = np.where(
        df["oldbalanceOrg"] > 0,
        df["amount"] / df["oldbalanceOrg"],
        np.nan
    )

    # 4. Whether sender had zero balance
    df["zero_sender_balance"] = (
        df["oldbalanceOrg"] == 0
    ).astype(int)

    return df