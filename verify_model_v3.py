import mlflow
import pandas as pd


MODEL_URI = "models:/fraud_detection_model/3"


transaction = pd.DataFrame(
    [
        {
            "step": 500,
            "type": "TRANSFER",
            "amount": 500000,
            "oldbalanceOrg": 500000,
            "newbalanceOrig": 0,
            "oldbalanceDest": 100000,
            "newbalanceDest": 600000
        }
    ]
)


print("Loading Version 3...")

model = mlflow.pyfunc.load_model(
    MODEL_URI
)


print(
    "Version 3 loaded successfully."
)


print(
    "\nRunning prediction..."
)


prediction = model.predict(
    transaction
)


print(
    "\nFraud probability:",
    prediction
)