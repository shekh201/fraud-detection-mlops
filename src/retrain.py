import mlflow
import mlflow.pyfunc
import mlflow.xgboost

import pandas as pd
import xgboost as xgb
import yaml

from mlflow import MlflowClient

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

from src.features import create_features
from src.preprocessing import create_preprocessor
from src.mlflow_model import FraudDetectionModel


# ============================================================
# 1. Configuration
# ============================================================

CONFIG_PATH = "configs/retraining.yml"

DATA_PATH = "data/raw/paysim.csv"


RAW_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud"
]


FEATURE_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balance_change",
    "balance_error",
    "amount_balance_ratio",
    "zero_sender_balance"
]


PREDICTION_THRESHOLD = 0.99


# ============================================================
# 2. Load retraining configuration
# ============================================================

with open(
    CONFIG_PATH,
    "r"
) as file:

    config = yaml.safe_load(file)


retraining_config = config["retraining"]


if not retraining_config["enabled"]:

    print(
        "Automated retraining is disabled."
    )

    raise SystemExit(0)


# ============================================================
# 3. Read configuration values
# ============================================================

MIN_TRAINING_ROWS = retraining_config[
    "min_training_rows"
]

PR_AUC_THRESHOLD = retraining_config[
    "validation_pr_auc_threshold"
]

RECALL_THRESHOLD = retraining_config[
    "validation_recall_threshold"
]

PRECISION_THRESHOLD = retraining_config[
    "validation_precision_threshold"
]

MLFLOW_EXPERIMENT = retraining_config[
    "mlflow_experiment"
]

REGISTERED_MODEL_NAME = retraining_config[
    "registered_model_name"
]

CHAMPION_ALIAS = retraining_config[
    "champion_alias"
]


# ============================================================
# 4. Evaluate current champion
# ============================================================

def evaluate_champion(
    model_name,
    champion_alias,
    X_validation_processed,
    X_validation_raw,
    y_validation
):

    try:

        champion_model_uri = (
            f"models:/{model_name}@{champion_alias}"
        )

        print(
            "\nLoading current champion model..."
        )

        # ----------------------------------------------------
        # First try the old XGBoost format.
        #
        # Version 2 is currently an XGBoost model that expects
        # already-preprocessed data.
        # ----------------------------------------------------

        try:

            champion_model = (
                mlflow.xgboost.load_model(
                    champion_model_uri
                )
            )

            champion_probability = (
                champion_model.predict_proba(
                    X_validation_processed
                )[:, 1]
            )

            print(
                "Champion format: XGBoost"
            )

        except Exception:

            # ------------------------------------------------
            # Future champion versions will use our custom
            # PyFunc package containing:
            #
            # create_features()
            # preprocessor
            # XGBoost
            # ------------------------------------------------

            champion_model = (
                mlflow.pyfunc.load_model(
                    champion_model_uri
                )
            )

            champion_probability = (
                champion_model.predict(
                    X_validation_raw
                )
            )

            print(
                "Champion format: MLflow PyFunc"
            )


        champion_prediction = (
            champion_probability >= PREDICTION_THRESHOLD
        ).astype(int)


        champion_pr_auc = average_precision_score(
            y_validation,
            champion_probability
        )


        champion_precision = precision_score(
            y_validation,
            champion_prediction,
            zero_division=0
        )


        champion_recall = recall_score(
            y_validation,
            champion_prediction,
            zero_division=0
        )


        champion_f1 = f1_score(
            y_validation,
            champion_prediction,
            zero_division=0
        )


        print(
            "\nChampion Metrics"
        )

        print(
            "----------------"
        )

        print(
            "PR-AUC:",
            champion_pr_auc
        )

        print(
            "Precision:",
            champion_precision
        )

        print(
            "Recall:",
            champion_recall
        )

        print(
            "F1:",
            champion_f1
        )


        return {
            "pr_auc": champion_pr_auc,
            "precision": champion_precision,
            "recall": champion_recall,
            "f1": champion_f1
        }


    except Exception as error:

        print(
            "\nChampion model could not be evaluated."
        )

        print(
            "Reason:",
            error
        )

        return None


# ============================================================
# 5. Load dataset
# ============================================================

print(
    "\nLoading dataset..."
)

df = pd.read_csv(
    DATA_PATH,
    usecols=RAW_COLUMNS
)

print(
    "Dataset rows:",
    len(df)
)


# ============================================================
# 6. Minimum data safety check
# ============================================================

if len(df) < MIN_TRAINING_ROWS:

    raise ValueError(
        f"Not enough training data. "
        f"Required: {MIN_TRAINING_ROWS}, "
        f"Available: {len(df)}"
    )


# ============================================================
# 7. Feature engineering
# ============================================================

print(
    "\nCreating features..."
)

df = create_features(
    df
)


# ============================================================
# 8. Time-based train/validation/test split
# ============================================================

print(
    "\nCreating time-based splits..."
)

train = df[
    df["step"] <= 520
].copy()


validation = df[
    (df["step"] > 520)
    &
    (df["step"] <= 631)
].copy()


test = df[
    df["step"] > 631
].copy()


print(
    "Train rows:",
    len(train)
)

print(
    "Validation rows:",
    len(validation)
)

print(
    "Test rows:",
    len(test)
)


# ============================================================
# 9. Separate features and target
# ============================================================

X_train = train[
    FEATURE_COLUMNS
]

y_train = train[
    "isFraud"
]


X_validation = validation[
    FEATURE_COLUMNS
]

y_validation = validation[
    "isFraud"
]


X_test = test[
    FEATURE_COLUMNS
]

y_test = test[
    "isFraud"
]


# ============================================================
# 10. Create preprocessing pipeline
# ============================================================

print(
    "\nCreating preprocessing pipeline..."
)

preprocessor = create_preprocessor()


# ============================================================
# 11. Fit preprocessing ONLY on training data
# ============================================================

print(
    "\nFitting preprocessor..."
)

X_train_processed = preprocessor.fit_transform(
    X_train
)


X_validation_processed = preprocessor.transform(
    X_validation
)


X_test_processed = preprocessor.transform(
    X_test
)


# ============================================================
# 12. Convert processed data to float32
# ============================================================

X_train_processed = X_train_processed.astype(
    "float32"
)

X_validation_processed = X_validation_processed.astype(
    "float32"
)

X_test_processed = X_test_processed.astype(
    "float32"
)


print(
    "Processed train shape:",
    X_train_processed.shape
)

print(
    "Processed validation shape:",
    X_validation_processed.shape
)

print(
    "Processed test shape:",
    X_test_processed.shape
)


# ============================================================
# 13. Calculate class imbalance weight
# ============================================================

negative_count = (
    y_train == 0
).sum()


positive_count = (
    y_train == 1
).sum()


scale_pos_weight = (
    negative_count /
    positive_count
)


print(
    "\nscale_pos_weight:",
    scale_pos_weight
)


# ============================================================
# 14. Create XGBoost model
# ============================================================

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


# ============================================================
# 15. Configure MLflow
# ============================================================

mlflow.set_experiment(
    MLFLOW_EXPERIMENT
)


# ============================================================
# 16. Start MLflow run
# ============================================================

with mlflow.start_run(
    run_name="automated-retraining-packaged"
):

    print(
        "\nStarting model training..."
    )

    model.fit(
        X_train_processed,
        y_train
    )


    # ========================================================
    # 17. Generate validation predictions
    # ========================================================

    validation_probability = (
        model.predict_proba(
            X_validation_processed
        )[:, 1]
    )


    validation_prediction = (
        validation_probability >= PREDICTION_THRESHOLD
    ).astype(int)


    # ========================================================
    # 18. Calculate validation metrics
    # ========================================================

    validation_pr_auc = (
        average_precision_score(
            y_validation,
            validation_probability
        )
    )


    validation_precision = (
        precision_score(
            y_validation,
            validation_prediction,
            zero_division=0
        )
    )


    validation_recall = (
        recall_score(
            y_validation,
            validation_prediction,
            zero_division=0
        )
    )


    validation_f1 = (
        f1_score(
            y_validation,
            validation_prediction,
            zero_division=0
        )
    )


    # ========================================================
    # 19. Generate test predictions
    # ========================================================

    test_probability = (
        model.predict_proba(
            X_test_processed
        )[:, 1]
    )


    test_prediction = (
        test_probability >= PREDICTION_THRESHOLD
    ).astype(int)


    # ========================================================
    # 20. Calculate test metrics
    # ========================================================

    test_pr_auc = (
        average_precision_score(
            y_test,
            test_probability
        )
    )


    test_precision = (
        precision_score(
            y_test,
            test_prediction,
            zero_division=0
        )
    )


    test_recall = (
        recall_score(
            y_test,
            test_prediction,
            zero_division=0
        )
    )


    test_f1 = (
        f1_score(
            y_test,
            test_prediction,
            zero_division=0
        )
    )


    # ========================================================
    # 21. Print metrics
    # ========================================================

    print(
        "\nValidation Metrics"
    )

    print(
        "------------------"
    )

    print(
        "PR-AUC:",
        validation_pr_auc
    )

    print(
        "Precision:",
        validation_precision
    )

    print(
        "Recall:",
        validation_recall
    )

    print(
        "F1:",
        validation_f1
    )


    print(
        "\nTest Metrics"
    )

    print(
        "------------"
    )

    print(
        "PR-AUC:",
        test_pr_auc
    )

    print(
        "Precision:",
        test_precision
    )

    print(
        "Recall:",
        test_recall
    )

    print(
        "F1:",
        test_f1
    )


    # ========================================================
    # 22. Log parameters
    # ========================================================

    mlflow.log_params(
        {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "threshold": PREDICTION_THRESHOLD,
            "scale_pos_weight": scale_pos_weight,
            "model_format": "mlflow_pyfunc",
            "preprocessor_packaged": True
        }
    )


    # ========================================================
    # 23. Log metrics
    # ========================================================

    mlflow.log_metrics(
        {
            "validation_pr_auc": validation_pr_auc,
            "validation_precision": validation_precision,
            "validation_recall": validation_recall,
            "validation_f1": validation_f1,
            "test_pr_auc": test_pr_auc,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1
        }
    )


    # ========================================================
    # 24. Minimum threshold check
    # ========================================================

    minimum_thresholds_passed = (
        validation_pr_auc >= PR_AUC_THRESHOLD
        and
        validation_recall >= RECALL_THRESHOLD
        and
        validation_precision >= PRECISION_THRESHOLD
    )


    print(
        "\nMinimum Threshold Check"
    )

    print(
        "-----------------------"
    )

    print(
        "PR-AUC threshold:",
        validation_pr_auc >= PR_AUC_THRESHOLD
    )

    print(
        "Recall threshold:",
        validation_recall >= RECALL_THRESHOLD
    )

    print(
        "Precision threshold:",
        validation_precision >= PRECISION_THRESHOLD
    )


    # ========================================================
    # 25. Compare with current champion
    # ========================================================

    champion_metrics = None

    new_model_is_better = False


    if minimum_thresholds_passed:

        champion_metrics = evaluate_champion(
            model_name=REGISTERED_MODEL_NAME,
            champion_alias=CHAMPION_ALIAS,
            X_validation_processed=X_validation_processed,
            X_validation_raw=X_validation,
            y_validation=y_validation
        )


    # ========================================================
    # 26. Champion comparison
    # ========================================================

    if champion_metrics is not None:

        new_model_is_better = (
            validation_pr_auc >= champion_metrics["pr_auc"]
            and
            validation_recall >= champion_metrics["recall"]
            and
            validation_precision >= champion_metrics["precision"]
        )


    # ========================================================
    # 27. Final model acceptance
    # ========================================================

    model_accepted = (
        minimum_thresholds_passed
        and
        new_model_is_better
    )


    print(
        "\nModel Acceptance"
    )

    print(
        "----------------"
    )

    print(
        "Minimum thresholds passed:",
        minimum_thresholds_passed
    )

    print(
        "Champion comparison passed:",
        new_model_is_better
    )

    print(
        "Accepted:",
        model_accepted
    )


    # ========================================================
    # 28. Register packaged model
    # ========================================================

    if model_accepted:

        print(
            "\nModel passed validation."
        )

        print(
            "Creating packaged MLflow model..."
        )


        packaged_model = FraudDetectionModel(
            model=model,
            preprocessor=preprocessor
        )


        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=packaged_model,
            code_paths=["src"],
            registered_model_name=REGISTERED_MODEL_NAME
        )


        print(
            "Packaged model logged."
        )

        print(
            "Model URI:",
            model_info.model_uri
        )


        # ----------------------------------------------------
        # Find newly created model version
        # ----------------------------------------------------

        client = MlflowClient()


        latest_versions = (
            client.search_model_versions(
                f"name='{REGISTERED_MODEL_NAME}'"
            )
        )


        latest_version = max(
            latest_versions,
            key=lambda version: int(
                version.version
            )
        )


        # ----------------------------------------------------
        # Add metadata to registered model version
        # ----------------------------------------------------

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            latest_version.version,
            "model_format",
            "mlflow_pyfunc"
        )


        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            latest_version.version,
            "preprocessor_packaged",
            "true"
        )


        # ----------------------------------------------------
        # Set champion alias
        # ----------------------------------------------------

        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME,
            CHAMPION_ALIAS,
            latest_version.version
        )


        print(
            "\nModel registered successfully."
        )

        print(
            "Champion version:",
            latest_version.version
        )

        print(
            "Model format: MLflow PyFunc"
        )

        print(
            "Preprocessor packaged: True"
        )


    else:

        print(
            "\nModel rejected."
        )

        print(
            "Model will NOT be registered."
        )