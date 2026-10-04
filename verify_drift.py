import pandas as pd

from src.features import create_features
from src.drift import (
    calculate_psi,
    calculate_categorical_psi,
    interpret_psi
)

from src.metrics import (
    FEATURE_DRIFT_PSI,
    FEATURE_DRIFT_STATUS
)


RAW_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


df = pd.read_csv(
    "data/raw/paysim.csv",
    usecols=RAW_COLUMNS
)


df = create_features(df)


train = df[df["step"] <= 520]

test = df[df["step"] > 631]


features_to_monitor = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balance_change",
    "balance_error",
    "amount_balance_ratio",
    "zero_sender_balance",
]


def drift_status_to_number(status):

    if status == "LOW":
        return 0

    if status == "MEDIUM":
        return 1

    return 2


sample_size = 100_000


train_sample = train.sample(
    n=min(sample_size, len(train)),
    random_state=42
)


test_sample = test.sample(
    n=min(sample_size, len(test)),
    random_state=42
)


print("\nData sizes:")
print("Train:", len(train))
print("Test :", len(test))


print("\nSample sizes:")
print("Train sample:", len(train_sample))
print("Test sample :", len(test_sample))


print("\nDrift Results:")
print("-" * 65)


# Numerical feature drift
for feature in features_to_monitor:

    psi = calculate_psi(
        train_sample[feature],
        test_sample[feature]
    )

    status = interpret_psi(psi)

    status_number = drift_status_to_number(status)

    FEATURE_DRIFT_PSI.labels(
        feature=feature
    ).set(psi)

    FEATURE_DRIFT_STATUS.labels(
        feature=feature
    ).set(status_number)

    print(
        f"{feature:25s} "
        f"PSI={psi:.4f} "
        f"DRIFT={status}"
    )


# Categorical feature drift
type_psi = calculate_categorical_psi(
    train_sample["type"],
    test_sample["type"]
)

type_status = interpret_psi(type_psi)

type_status_number = drift_status_to_number(
    type_status
)

FEATURE_DRIFT_PSI.labels(
    feature="type"
).set(type_psi)

FEATURE_DRIFT_STATUS.labels(
    feature="type"
).set(type_status_number)

print(
    f"{'type':25s} "
    f"PSI={type_psi:.4f} "
    f"DRIFT={type_status}"
)