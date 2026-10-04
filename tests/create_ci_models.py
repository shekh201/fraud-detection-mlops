import os

import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest

from src.features import create_features
from src.mlflow_model import FraudDetectionModel
from src.preprocessing import create_preprocessor


MODEL_DIR = "models"


def create_training_data():
    np.random.seed(42)

    rows = 200

    data = pd.DataFrame({
        "step": np.random.randint(1, 1000, rows),
        "type": np.random.choice(
            [
                "PAYMENT",
                "TRANSFER",
                "CASH_OUT",
                "CASH_IN",
                "DEBIT"
            ],
            rows
        ),
        "amount": np.random.uniform(
            100,
            1_000_000,
            rows
        ),
        "oldbalanceOrg": np.random.uniform(
            1_000,
            2_000_000,
            rows
        ),
        "oldbalanceDest": np.random.uniform(
            1_000,
            2_000_000,
            rows
        ),
    })

    data["newbalanceOrig"] = (
        data["oldbalanceOrg"]
        - data["amount"]
    ).clip(lower=0)

    data["newbalanceDest"] = (
        data["oldbalanceDest"]
        + data["amount"]
    )

    data["isFraud"] = (
        (
            (data["type"] == "TRANSFER")
            | (data["type"] == "CASH_OUT")
        )
        & (data["amount"] > 500_000)
    ).astype(int)

    return data


def main():

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    data = create_training_data()

    features = data.drop(
        columns=["isFraud"]
    )

    target = data["isFraud"]

    features = create_features(
        features
    )

    preprocessor = create_preprocessor()

    processed_features = preprocessor.fit_transform(
        features
    )

    processed_features = processed_features.astype(
        np.float32
    )

    fraud_model = xgb.XGBClassifier(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1
    )

    fraud_model.fit(
        processed_features,
        target
    )

    anomaly_model = IsolationForest(
        n_estimators=20,
        contamination="auto",
        random_state=42,
        n_jobs=1
    )

    anomaly_model.fit(
        processed_features
    )

    joblib.dump(
        preprocessor,
        os.path.join(
            MODEL_DIR,
            "preprocessor.joblib"
        )
    )

    joblib.dump(
        anomaly_model,
        os.path.join(
            MODEL_DIR,
            "anomaly_model.joblib"
        )
    )

    packaged_model = FraudDetectionModel(
        model=fraud_model,
        preprocessor=preprocessor
    )

    mlflow_model_path = os.path.join(
        MODEL_DIR,
        "ci_mlflow_champion"
    )

    mlflow.pyfunc.save_model(
        path=mlflow_model_path,
        python_model=packaged_model,
        code_paths=["src"]
    )

    print("CI test models created successfully")

    print(
        "Fraud model:",
        mlflow_model_path
    )

    print(
        "Preprocessor:",
        os.path.join(
            MODEL_DIR,
            "preprocessor.joblib"
        )
    )

    print(
        "Anomaly model:",
        os.path.join(
            MODEL_DIR,
            "anomaly_model.joblib"
        )
    )

if __name__ == "__main__":
    main()