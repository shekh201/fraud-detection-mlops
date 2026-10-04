import os

import psycopg


def get_database_connection():

    connection = psycopg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "frauddb"),
        user=os.getenv("DB_USER", "frauduser"),
        password=os.getenv("DB_PASSWORD", "fraudpassword")
    )

    return connection


def save_prediction(
    transaction,
    prediction
):

    connection = get_database_connection()

    try:

        with connection.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO transactions (
                    step,
                    type,
                    amount,
                    old_balance_org,
                    new_balance_orig,
                    old_balance_dest,
                    new_balance_dest,
                    fraud_probability,
                    risk_decision,
                    anomaly_score
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )
                """,
                (
                    transaction["step"],
                    transaction["type"],
                    transaction["amount"],
                    transaction["oldbalanceOrg"],
                    transaction["newbalanceOrig"],
                    transaction["oldbalanceDest"],
                    transaction["newbalanceDest"],
                    prediction["fraud_probability"],
                    prediction["risk_decision"],
                    prediction["anomaly_score"]
                )
            )

        connection.commit()

    finally:

        connection.close()
        
def get_predictions(
    limit=20,
    offset=0,
    risk_decision=None,
    transaction_type=None
):

    connection = get_database_connection()

    try:

        with connection.cursor() as cursor:

            query = """
                SELECT
                    id,
                    created_at,
                    step,
                    type,
                    amount,
                    old_balance_org,
                    new_balance_orig,
                    old_balance_dest,
                    new_balance_dest,
                    fraud_probability,
                    risk_decision,
                    anomaly_score
                FROM transactions
                WHERE 1=1
            """

            parameters = []

            # --------------------------------------
            # Risk decision filter
            # --------------------------------------

            if risk_decision is not None:

                query += """
                    AND risk_decision = %s
                """

                parameters.append(
                    risk_decision
                )

            # --------------------------------------
            # Transaction type filter
            # --------------------------------------

            if transaction_type is not None:

                query += """
                    AND type = %s
                """

                parameters.append(
                    transaction_type
                )

            # --------------------------------------
            # Ordering
            # --------------------------------------

            query += """
                ORDER BY id DESC
                LIMIT %s
                OFFSET %s
            """

            parameters.extend(
                [limit, offset]
            )

            cursor.execute(
                query,
                parameters
            )

            rows = cursor.fetchall()

        return rows

    finally:

        connection.close()
        
def get_prediction_count(
    risk_decision=None,
    transaction_type=None
):

    connection = get_database_connection()

    try:

        with connection.cursor() as cursor:

            query = """
                SELECT COUNT(*)
                FROM transactions
                WHERE 1=1
            """

            parameters = []

            if risk_decision is not None:

                query += """
                    AND risk_decision = %s
                """

                parameters.append(
                    risk_decision
                )

            if transaction_type is not None:

                query += """
                    AND type = %s
                """

                parameters.append(
                    transaction_type
                )

            cursor.execute(
                query,
                parameters
            )

            result = cursor.fetchone()

            return result[0]

    finally:

        connection.close()