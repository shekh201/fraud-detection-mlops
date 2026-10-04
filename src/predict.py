import numpy as np
import pandas as pd

from src.features import create_features
from src.risk_engine import get_risk_decision


def predict_transaction(
    transaction,
    model,
    preprocessor,
    anomaly_model
):
    
    transaction_df = pd.DataFrame([transaction])

    transaction_df = create_features(
        transaction_df
    )

    processed_transaction = preprocessor.transform(
        transaction_df
    )

    processed_transaction = (
        processed_transaction.astype(np.float32)
    )

    fraud_probability = model.predict_proba(
        processed_transaction
    )[0, 1]

    risk_decision = get_risk_decision(
        fraud_probability
    )

    anomaly_score = -anomaly_model.decision_function(
        processed_transaction
    )[0]

    return {
        "fraud_probability": float(fraud_probability),
        "risk_decision": risk_decision,
        "anomaly_score": float(anomaly_score)
    }