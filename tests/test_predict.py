import joblib
import xgboost as xgb

from src.predict import predict_transaction
from src.preprocessing import create_preprocessor


def test_prediction_pipeline():
    transaction = {
        "step": 501,
        "type": "TRANSFER",
        "amount": 123456.0,
        "oldbalanceOrg": 123456.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 200000.0,
        "newbalanceDest": 323456.0,
    }

    model = xgb.XGBClassifier()
    model.load_model("models/fraud_model.json")

    preprocessor = joblib.load(
        "models/preprocessor.joblib"
    )

    anomaly_model = joblib.load(
        "models/anomaly_model.joblib"
    )

    result = predict_transaction(
        transaction,
        model,
        preprocessor,
        anomaly_model
    )

    assert "fraud_probability" in result
    assert "risk_decision" in result
    assert "anomaly_score" in result

    assert 0.0 <= result["fraud_probability"] <= 1.0

    assert result["risk_decision"] in {
        "ALLOW",
        "REVIEW",
        "BLOCK"
    }