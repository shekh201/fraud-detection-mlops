from mlflow import MlflowClient


client = MlflowClient()


MODEL_NAME = "fraud_detection_model"
CHAMPION_ALIAS = "champion"
VERSION = "2"


client.set_registered_model_alias(
    MODEL_NAME,
    CHAMPION_ALIAS,
    VERSION
)


print(
    f"{MODEL_NAME} version {VERSION} "
    f"is now assigned to @{CHAMPION_ALIAS}"
)