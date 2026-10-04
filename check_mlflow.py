from mlflow import MlflowClient


client = MlflowClient()


MODEL_NAME = "fraud_detection_model"


model = client.get_registered_model(
    MODEL_NAME
)


print("Registered Model:")
print(model.name)


print("\nModel Aliases:")
print(model.aliases)


print("\nVersions:")


for version in client.search_model_versions(
    f"name='{MODEL_NAME}'"
):

    print(
        "Version:",
        version.version
    )

    print(
        "Status:",
        version.status
    )

    print(
        "Source:",
        version.source
    )

    print(
        "Aliases:",
        version.aliases
    )

    print(
        "Tags:",
        version.tags
    )

    print(
        "-------------------------"
    )