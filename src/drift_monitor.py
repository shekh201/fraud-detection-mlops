import pandas as pd

from src.features import create_features

from src.drift import (
    calculate_psi,
    calculate_categorical_psi,
    interpret_psi
)

from src.database import get_database_connection

from src.metrics import (
    FEATURE_DRIFT_PSI,
    FEATURE_DRIFT_STATUS,
    DRIFT_MONITOR_STATUS
)


REFERENCE_PATH = "models/drift_reference.parquet"

CURRENT_SAMPLE_SIZE = 10_000

MIN_CURRENT_ROWS = 1000


FEATURES_TO_MONITOR = [
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


def drift_status_to_number(status):

    if status == "LOW":
        return 0

    if status == "MEDIUM":
        return 1

    return 2


def load_reference_data():

    reference = pd.read_parquet(
        REFERENCE_PATH
    )

    return reference


def load_current_data():

    connection = get_database_connection()

    try:

        query = """
            SELECT
                step,
                type,
                amount,
                old_balance_org,
                new_balance_orig,
                old_balance_dest,
                new_balance_dest
            FROM transactions
            ORDER BY id DESC
            LIMIT %s
        """

        with connection.cursor() as cursor:

            cursor.execute(
                query,
                (CURRENT_SAMPLE_SIZE,)
            )

            rows = cursor.fetchall()

        columns = [
            "step",
            "type",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest"
        ]

        current = pd.DataFrame(
            rows,
            columns=columns
        )

        return current

    finally:

        connection.close()


def update_drift_metrics():

    reference = load_reference_data()

    current = load_current_data()

    current_rows = len(current)

    if current_rows < MIN_CURRENT_ROWS:
        
        DRIFT_MONITOR_STATUS.set(1)

        return {
            "status": "INSUFFICIENT_DATA",
            "current_rows": current_rows,
            "required_rows": MIN_CURRENT_ROWS,
            "message": (
                "Not enough production transactions "
                "to calculate reliable drift"
            )
        }


    current = create_features(
        current
    )


    results = {}


    for feature in FEATURES_TO_MONITOR:

        psi = calculate_psi(
            reference[feature],
            current[feature]
        )

        status = interpret_psi(
            psi
        )

        status_number = drift_status_to_number(
            status
        )


        FEATURE_DRIFT_PSI.labels(
            feature=feature
        ).set(psi)


        FEATURE_DRIFT_STATUS.labels(
            feature=feature
        ).set(status_number)


        results[feature] = {
            "psi": psi,
            "status": status
        }


    type_psi = calculate_categorical_psi(
        reference["type"],
        current["type"]
    )


    type_status = interpret_psi(
        type_psi
    )


    type_status_number = drift_status_to_number(
        type_status
    )


    FEATURE_DRIFT_PSI.labels(
        feature="type"
    ).set(type_psi)


    FEATURE_DRIFT_STATUS.labels(
        feature="type"
    ).set(type_status_number)


    results["type"] = {
        "psi": type_psi,
        "status": type_status
    }

    DRIFT_MONITOR_STATUS.set(0)
    
    return {
        "status": "SUCCESS",
        "current_rows": current_rows,
        "results": results
    }