import joblib
import mlflow
import pandas as pd

from src.features import create_features


# ============================================================
# 1. Model versions
# ============================================================

MODEL_NAME = "fraud_detection_model"

V2_URI = f"models:/{MODEL_NAME}/2"

V3_URI = f"models:/{MODEL_NAME}/3"


# ============================================================
# 2. Load existing preprocessor
# ============================================================

print(
    "Loading existing preprocessor..."
)

preprocessor = joblib.load(
    "models/preprocessor.joblib"
)

print(
    "Preprocessor loaded successfully."
)


# ============================================================
# 3. Test transactions
# ============================================================

transactions = pd.DataFrame(
    [
        {
            "step": 500,
            "type": "TRANSFER",
            "amount": 500000,
            "oldbalanceOrg": 500000,
            "newbalanceOrig": 0,
            "oldbalanceDest": 100000,
            "newbalanceDest": 600000
        },
        {
            "step": 600,
            "type": "PAYMENT",
            "amount": 1000,
            "oldbalanceOrg": 50000,
            "newbalanceOrig": 49000,
            "oldbalanceDest": 10000,
            "newbalanceDest": 11000
        },
        {
            "step": 700,
            "type": "CASH_OUT",
            "amount": 800000,
            "oldbalanceOrg": 800000,
            "newbalanceOrig": 0,
            "oldbalanceDest": 200000,
            "newbalanceDest": 1000000
        }
    ]
)


# ============================================================
# 4. Create features for Version 2
# ============================================================

transactions_with_features = create_features(
    transactions
)


# ============================================================
# 5. Process input for Version 2
# ============================================================

processed_transactions = (
    preprocessor.transform(
        transactions_with_features
    )
)


# ============================================================
# 6. Load Version 2
# ============================================================

print(
    "\nLoading Version 2..."
)

model_v2 = mlflow.xgboost.load_model(
    V2_URI
)

print(
    "Version 2 loaded successfully."
)


# ============================================================
# 7. Load Version 3
# ============================================================

print(
    "\nLoading Version 3..."
)

model_v3 = mlflow.pyfunc.load_model(
    V3_URI
)

print(
    "Version 3 loaded successfully."
)


# ============================================================
# 8. Generate Version 2 predictions
# ============================================================

predictions_v2 = (
    model_v2.predict_proba(
        processed_transactions
    )[:, 1]
)


# ============================================================
# 9. Generate Version 3 predictions
# ============================================================

predictions_v3 = model_v3.predict(
    transactions
)


# ============================================================
# 10. Compare predictions
# ============================================================

print(
    "\nPrediction Comparison"
)

print(
    "====================="
)


for index in range(
    len(transactions)
):

    v2_probability = predictions_v2[
        index
    ]

    v3_probability = predictions_v3[
        index
    ]

    difference = abs(
        v2_probability -
        v3_probability
    )


    print(
        f"\nTransaction {index + 1}"
    )

    print(
        "Type:",
        transactions.iloc[index]["type"]
    )

    print(
        "V2 probability:",
        v2_probability
    )

    print(
        "V3 probability:",
        v3_probability
    )

    print(
        "Difference:",
        difference
    )


# ============================================================
# 11. Calculate maximum difference
# ============================================================

max_difference = max(
    abs(
        predictions_v2 -
        predictions_v3
    )
)


print(
    "\nMaximum probability difference:",
    max_difference
)


# ============================================================
# 12. Final consistency check
# ============================================================

if max_difference < 0.0001:

    print(
        "\nPASS: V2 and V3 predictions are consistent."
    )

else:

    print(
        "\nWARNING: V2 and V3 predictions differ."
    )