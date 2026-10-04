import pandas as pd

from src.preprocessing import create_preprocessor


def create_sample_data():
    return pd.DataFrame({
        "step": [1, 2, 3, 4, 5],
        "type": [
            "PAYMENT",
            "TRANSFER",
            "CASH_OUT",
            "CASH_IN",
            "DEBIT"
        ],
        "amount": [100.0, 200.0, 300.0, 400.0, 500.0],
        "oldbalanceOrg": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
        "newbalanceOrig": [900.0, 1800.0, 2700.0, 3600.0, 4500.0],
        "oldbalanceDest": [500.0, 600.0, 700.0, 800.0, 900.0],
        "newbalanceDest": [600.0, 800.0, 1000.0, 1200.0, 1400.0],
        "balance_change": [100.0, 200.0, 300.0, 400.0, 500.0],
        "balance_error": [0.0, 0.0, 0.0, 0.0, 0.0],
        "amount_balance_ratio": [0.1, 0.1, 0.1, 0.1, 0.1],
        "zero_sender_balance": [0, 0, 0, 0, 0],
    })


def test_preprocessor_fits_and_transforms():
    df = create_sample_data()

    preprocessor = create_preprocessor()

    result = preprocessor.fit_transform(df)

    assert result.shape[0] == 5
    assert result.shape[1] == 15


def test_preprocessor_handles_missing_numeric_value():
    df = create_sample_data()

    df.loc[1, "amount"] = None

    preprocessor = create_preprocessor()

    result = preprocessor.fit_transform(df)

    assert result.shape == (5, 15)


def test_preprocessor_handles_unknown_category():
    train_df = create_sample_data()

    preprocessor = create_preprocessor()

    preprocessor.fit(train_df)

    new_data = create_sample_data()

    new_data.loc[0, "type"] = "NEW_TRANSACTION_TYPE"

    result = preprocessor.transform(new_data)

    assert result.shape == (5, 15)