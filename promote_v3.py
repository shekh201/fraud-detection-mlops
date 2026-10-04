from mlflow import MlflowClient


# ============================================================
# 1. MLflow client
# ============================================================

client = MlflowClient()


# ============================================================
# 2. Model configuration
# ============================================================

MODEL_NAME = "fraud_detection_model"

CHAMPION_ALIAS = "champion"

NEW_CHAMPION_VERSION = "3"


# ============================================================
# 3. Promote Version 3
# ============================================================

client.set_registered_model_alias(
    MODEL_NAME,
    CHAMPION_ALIAS,
    NEW_CHAMPION_VERSION
)


# ============================================================
# 4. Confirmation
# ============================================================

print(
    f"{MODEL_NAME} version "
    f"{NEW_CHAMPION_VERSION} "
    f"is now @{CHAMPION_ALIAS}"
)