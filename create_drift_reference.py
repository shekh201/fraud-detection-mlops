import os

import pandas as pd

from src.features import create_features


RAW_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


REFERENCE_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balance_change",
    "balance_error",
    "amount_balance_ratio",
    "zero_sender_balance",
    "type",
]


REFERENCE_SAMPLE_SIZE = 100_000


df = pd.read_csv(
    "data/raw/paysim.csv",
    usecols=RAW_COLUMNS
)


df = create_features(df)


train = df[df["step"] <= 520]


reference_sample = train.sample(
    n=min(
        REFERENCE_SAMPLE_SIZE,
        len(train)
    ),
    random_state=42
)


reference = reference_sample[
    REFERENCE_FEATURES
].copy()


os.makedirs(
    "models",
    exist_ok=True
)


reference.to_parquet(
    "models/drift_reference.parquet",
    index=False
)


print("\nDrift reference created successfully.")

print(
    "Reference rows:",
    len(reference)
)

print(
    "Reference columns:",
    list(reference.columns)
)

print(
    "\nSaved to:",
    "models/drift_reference.parquet"
)