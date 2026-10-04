import joblib
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from src.features import create_features
from src.preprocessing import create_preprocessor
from src.anomaly import create_anomaly_model


def setup_mlflow():

    mlflow.set_tracking_uri(
        "sqlite:///../mlflow.db"
    )

    mlflow.set_experiment(
        "fraud-detection-xgboost"
    )


def calculate_metrics(
    model,
    X_processed,
    y_true,
    threshold=0.99
):

    probabilities = model.predict_proba(
        X_processed
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    pr_auc = average_precision_score(
        y_true,
        probabilities
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions
    ).ravel()

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    return {
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


def load_training_data(data_path):

    df = pd.read_csv(data_path)

    df = create_features(df)

    X = df.drop(
        columns=[
            "isFraud",
            "isFlaggedFraud",
            "nameOrig",
            "nameDest"
        ]
    )

    y = df["isFraud"]

    return X, y


def split_data_by_time(X, y):

    train_mask = X["step"] <= 520

    validation_mask = (
        (X["step"] > 520)
        & (X["step"] <= 631)
    )

    test_mask = X["step"] > 631

    X_train = X.loc[train_mask].copy()
    y_train = y.loc[train_mask].copy()

    X_val = X.loc[validation_mask].copy()
    y_val = y.loc[validation_mask].copy()

    X_test = X.loc[test_mask].copy()
    y_test = y.loc[test_mask].copy()

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


def prepare_preprocessor(X_train):

    preprocessor = create_preprocessor()

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    return (
        preprocessor,
        X_train_processed
    )


def calculate_scale_pos_weight(y_train):

    negative_count = (
        y_train == 0
    ).sum()

    positive_count = (
        y_train == 1
    ).sum()

    scale_pos_weight = (
        negative_count / positive_count
    )

    return scale_pos_weight


def create_fraud_model(
    scale_pos_weight
):

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    return model


def train_fraud_model(
    model,
    X_train_processed,
    y_train,
    X_val_processed,
    y_val
):

    model.fit(
        X_train_processed,
        y_train,
        eval_set=[
            (X_val_processed, y_val)
        ],
        verbose=True
    )

    return model


def train_anomaly_model(
    X_train_processed,
    y_train,
    sample_size=100_000
):

    normal_indices = np.where(
        y_train.values == 0
    )[0]

    rng = np.random.default_rng(42)

    sample_indices = rng.choice(
        normal_indices,
        size=sample_size,
        replace=False
    )

    X_anomaly_train = (
        X_train_processed[sample_indices]
    )

    anomaly_model = create_anomaly_model()

    anomaly_model.fit(
        X_anomaly_train
    )

    return anomaly_model


def train_pipeline(data_path):

    # 1. Load data
    X, y = load_training_data(
        data_path
    )

    # 2. Time-based split
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_data_by_time(
        X,
        y
    )

    # 3. Fit preprocessor
    (
        preprocessor,
        X_train_processed
    ) = prepare_preprocessor(
        X_train
    )

    # 4. Transform validation and test
    X_val_processed = (
        preprocessor.transform(X_val)
    )

    X_test_processed = (
        preprocessor.transform(X_test)
    )

    # 5. Convert to float32
    X_train_processed = (
        X_train_processed.astype(
            np.float32
        )
    )

    X_val_processed = (
        X_val_processed.astype(
            np.float32
        )
    )

    X_test_processed = (
        X_test_processed.astype(
            np.float32
        )
    )

    # 6. Class imbalance weight
    scale_pos_weight = (
        calculate_scale_pos_weight(
            y_train
        )
    )

    # 7. Create XGBoost model
    fraud_model = create_fraud_model(
        scale_pos_weight
    )

    # 8. Train XGBoost
    fraud_model = train_fraud_model(
        fraud_model,
        X_train_processed,
        y_train,
        X_val_processed,
        y_val
    )

    # 9. Train anomaly model
    anomaly_model = train_anomaly_model(
        X_train_processed,
        y_train
    )

    return {
        "fraud_model": fraud_model,
        "anomaly_model": anomaly_model,
        "preprocessor": preprocessor,
        "X_val_processed": X_val_processed,
        "y_val": y_val,
        "X_test_processed": X_test_processed,
        "y_test": y_test
    }


def train_pipeline_with_mlflow(
    data_path
):

    setup_mlflow()

    with mlflow.start_run(
        run_name="xgboost-fraud-training"
    ):

        # -------------------------
        # 1. Log model parameters
        # -------------------------

        mlflow.log_param(
            "model",
            "XGBoost"
        )

        mlflow.log_param(
            "n_estimators",
            300
        )

        mlflow.log_param(
            "max_depth",
            6
        )

        mlflow.log_param(
            "learning_rate",
            0.1
        )

        mlflow.log_param(
            "subsample",
            0.8
        )

        mlflow.log_param(
            "colsample_bytree",
            0.8
        )

        # -------------------------
        # 2. Train models
        # -------------------------

        result = train_pipeline(
            data_path
        )

        # -------------------------
        # 3. Save models
        # -------------------------

        result["fraud_model"].save_model(
            "../models/fraud_model.json"
        )

        joblib.dump(
            result["preprocessor"],
            "../models/preprocessor.joblib"
        )

        joblib.dump(
            result["anomaly_model"],
            "../models/anomaly_model.joblib"
        )

        # -------------------------
        # 4. Calculate metrics
        # -------------------------

        validation_metrics = (
            calculate_metrics(
                result["fraud_model"],
                result["X_val_processed"],
                result["y_val"]
            )
        )

        test_metrics = (
            calculate_metrics(
                result["fraud_model"],
                result["X_test_processed"],
                result["y_test"]
            )
        )

        # -------------------------
        # 5. Log validation metrics
        # -------------------------

        mlflow.log_metric(
            "validation_pr_auc",
            validation_metrics["pr_auc"]
        )

        mlflow.log_metric(
            "validation_precision",
            validation_metrics["precision"]
        )

        mlflow.log_metric(
            "validation_recall",
            validation_metrics["recall"]
        )

        mlflow.log_metric(
            "validation_f1",
            validation_metrics["f1"]
        )

        # -------------------------
        # 6. Log test metrics
        # -------------------------

        mlflow.log_metric(
            "test_pr_auc",
            test_metrics["pr_auc"]
        )

        mlflow.log_metric(
            "test_precision",
            test_metrics["precision"]
        )

        mlflow.log_metric(
            "test_recall",
            test_metrics["recall"]
        )

        mlflow.log_metric(
            "test_f1",
            test_metrics["f1"]
        )

        # -------------------------
        # 7. Log artifacts
        # -------------------------

        mlflow.log_artifact(
            "../models/fraud_model.json",
            artifact_path="model"
        )

        mlflow.log_artifact(
            "../models/preprocessor.joblib",
            artifact_path="preprocessor"
        )

        mlflow.log_artifact(
            "../models/anomaly_model.joblib",
            artifact_path="anomaly_model"
        )

    return result