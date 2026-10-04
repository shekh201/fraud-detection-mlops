import pandas as pd

from src.features import create_features


def test_balance_change():
    df = pd.DataFrame({
        "oldbalanceOrg": [1000.0],
        "amount": [300.0],
        "newbalanceOrig": [700.0],
        "oldbalanceDest": [500.0],
        "newbalanceDest": [800.0],
    })

    result = create_features(df)

    assert result.loc[0, "balance_change"] == 300.0


def test_balance_error():
    df = pd.DataFrame({
        "oldbalanceOrg": [1000.0],
        "amount": [300.0],
        "newbalanceOrig": [700.0],
        "oldbalanceDest": [500.0],
        "newbalanceDest": [800.0],
    })

    result = create_features(df)

    assert result.loc[0, "balance_error"] == 0.0


def test_amount_balance_ratio():
    df = pd.DataFrame({
        "oldbalanceOrg": [1000.0],
        "amount": [250.0],
        "newbalanceOrig": [750.0],
        "oldbalanceDest": [500.0],
        "newbalanceDest": [750.0],
    })

    result = create_features(df)

    assert result.loc[0, "amount_balance_ratio"] == 0.25


def test_zero_sender_balance():
    df = pd.DataFrame({
        "oldbalanceOrg": [0.0],
        "amount": [500.0],
        "newbalanceOrig": [0.0],
        "oldbalanceDest": [1000.0],
        "newbalanceDest": [1500.0],
    })

    result = create_features(df)

    assert result.loc[0, "zero_sender_balance"] == 1
    
def test_amount_balance_ratio_when_balance_is_zero():
    df = pd.DataFrame({
        "oldbalanceOrg": [0.0],
        "amount": [500.0],
        "newbalanceOrig": [0.0],
        "oldbalanceDest": [1000.0],
        "newbalanceDest": [1500.0],
    })

    result = create_features(df)

    assert pd.isna(
        result.loc[0, "amount_balance_ratio"]
    )