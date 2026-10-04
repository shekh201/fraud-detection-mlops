import mlflow.pyfunc

from src.features import create_features


class FraudDetectionModel(
    mlflow.pyfunc.PythonModel
):

    def __init__(
        self,
        model,
        preprocessor
    ):
        self.model = model
        self.preprocessor = preprocessor


    def predict(
        self,
        context,
        model_input
    ):

        model_input = create_features(
            model_input
        )


        processed_input = (
            self.preprocessor.transform(
                model_input
            )
        )


        fraud_probability = (
            self.model.predict_proba(
                processed_input
            )[:, 1]
        )


        return fraud_probability