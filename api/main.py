import logging
import hashlib
import json
import asyncio

from contextlib import asynccontextmanager

import joblib
import mlflow
import pandas as pd

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Security,
    Response
)

from pydantic import BaseModel, Field

from typing import Literal


from src.drift_monitor import (
    update_drift_metrics
)


from src.database import (
    save_prediction,
    get_predictions,
    get_prediction_count
)


from src.cache import (
    get_prediction_cache,
    set_prediction_cache
)


from src.rate_limiter import (
    is_rate_limit_exceeded
)


from src.metrics import (
    PREDICTION_REQUESTS,
    PREDICTION_DECISIONS,
    PREDICTION_LATENCY,
    RATE_LIMIT_EXCEEDED,
    CACHE_HITS,
    CACHE_MISSES,
    DRIFT_MONITOR_STATUS
)


from src.auth import (
    verify_api_key
)


from src.features import (
    create_features
)


from src.risk_engine import (
    get_risk_decision
)


# ============================================================
# Logging configuration
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# Drift monitoring configuration
# ============================================================

DRIFT_CHECK_INTERVAL = 300


# ============================================================
# Background drift monitoring
# ============================================================

async def drift_monitor_loop():

    while True:

        try:

            result = update_drift_metrics()

            logger.info(
                "Drift monitor result: %s",
                result
            )

        except Exception:

            DRIFT_MONITOR_STATUS.set(2)

            logger.exception(
                "Drift monitoring failed"
            )

        await asyncio.sleep(
            DRIFT_CHECK_INTERVAL
        )


@asynccontextmanager
async def lifespan(app: FastAPI):

    drift_task = asyncio.create_task(
        drift_monitor_loop()
    )

    logger.info(
        "Drift monitoring background task started"
    )

    yield

    drift_task.cancel()

    try:

        await drift_task

    except asyncio.CancelledError:

        logger.info(
            "Drift monitoring background task stopped"
        )


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection API using MLflow champion model",
    version="2.0.0",
    lifespan=lifespan
)


# ============================================================
# Load MLflow Champion Fraud Model
# ============================================================

MLFLOW_MODEL_URI = (
    "models/mlflow_champion"
)


fraud_model = mlflow.pyfunc.load_model(
    MLFLOW_MODEL_URI
)


logger.info(
    "MLflow champion fraud model loaded successfully"
)


# ============================================================
# Load anomaly model
# ============================================================

anomaly_model = joblib.load(
    "models/anomaly_model.joblib"
)


logger.info(
    "Anomaly model loaded successfully"
)


# ============================================================
# Load preprocessor for anomaly model
# ============================================================

anomaly_preprocessor = joblib.load(
    "models/preprocessor.joblib"
)


logger.info(
    "Anomaly model preprocessor loaded successfully"
)


# ============================================================
# Request schema
# ============================================================

class Transaction(BaseModel):

    step: int = Field(
        ge=1,
        description="Transaction time step"
    )

    type: Literal[
        "PAYMENT",
        "TRANSFER",
        "CASH_OUT",
        "CASH_IN",
        "DEBIT"
    ]

    amount: float = Field(
        ge=0
    )

    oldbalanceOrg: float = Field(
        ge=0
    )

    newbalanceOrig: float = Field(
        ge=0
    )

    oldbalanceDest: float = Field(
        ge=0
    )

    newbalanceDest: float = Field(
        ge=0
    )


# ============================================================
# Prediction response schema
# ============================================================

class PredictionResponse(BaseModel):

    fraud_probability: float

    risk_decision: Literal[
        "ALLOW",
        "REVIEW",
        "BLOCK"
    ]

    anomaly_score: float


# ============================================================
# Transaction history response schema
# ============================================================

class TransactionResponse(BaseModel):

    id: int

    created_at: str

    step: int

    type: str

    amount: float

    old_balance_org: float

    new_balance_orig: float

    old_balance_dest: float

    new_balance_dest: float

    fraud_probability: float

    risk_decision: Literal[
        "ALLOW",
        "REVIEW",
        "BLOCK"
    ]

    anomaly_score: float


class TransactionListResponse(BaseModel):

    items: list[TransactionResponse]

    limit: int

    offset: int

    total: int


# ============================================================
# Redis cache key
# ============================================================

def create_cache_key(transaction):

    transaction_string = json.dumps(
        transaction,
        sort_keys=True
    )

    transaction_hash = hashlib.sha256(
        transaction_string.encode()
    ).hexdigest()

    return f"prediction:{transaction_hash}"


# ============================================================
# Root endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Fraud Detection API is running"
    }


# ============================================================
# Health endpoint
# ============================================================

@app.get("/health")
def health():

    logger.info(
        "Health check requested"
    )

    return {
        "status": "healthy",
        "model": "fraud_detection_model",
        "model_source": "MLflow @champion"
    }


# ============================================================
# Prometheus metrics endpoint
# ============================================================

@app.get("/metrics")
def metrics():

    from prometheus_client import (
        generate_latest,
        CONTENT_TYPE_LATEST
    )

    metrics_data = generate_latest()

    return Response(
        content=metrics_data,
        media_type=CONTENT_TYPE_LATEST
    )


# ============================================================
# Transaction history endpoint
# ============================================================

@app.get(
    "/transactions",
    response_model=TransactionListResponse
)
def get_transactions(
    limit: int = 20,
    offset: int = 0,
    risk_decision: Literal[
        "ALLOW",
        "REVIEW",
        "BLOCK"
    ] | None = None,
    transaction_type: Literal[
        "PAYMENT",
        "TRANSFER",
        "CASH_OUT",
        "CASH_IN",
        "DEBIT"
    ] | None = None,
    api_key: str = Security(verify_api_key)
):

    if limit < 1 or limit > 100:

        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 100"
        )

    if offset < 0:

        raise HTTPException(
            status_code=400,
            detail="offset cannot be negative"
        )

    try:

        rows = get_predictions(
            limit=limit,
            offset=offset,
            risk_decision=risk_decision,
            transaction_type=transaction_type
        )

        total = get_prediction_count(
            risk_decision=risk_decision,
            transaction_type=transaction_type
        )

        transactions = []

        for row in rows:

            transactions.append(
                {
                    "id": row[0],
                    "created_at": row[1].isoformat(),
                    "step": row[2],
                    "type": row[3],
                    "amount": row[4],
                    "old_balance_org": row[5],
                    "new_balance_orig": row[6],
                    "old_balance_dest": row[7],
                    "new_balance_dest": row[8],
                    "fraud_probability": row[9],
                    "risk_decision": row[10],
                    "anomaly_score": row[11]
                }
            )

        return {
            "items": transactions,
            "limit": limit,
            "offset": offset,
            "total": total
        }

    except Exception:

        logger.exception(
            "Failed to fetch transactions"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to fetch transactions"
        )


# ============================================================
# Prediction endpoint
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
@PREDICTION_LATENCY.time()
def predict(
    request: Request,
    transaction: Transaction,
    api_key: str = Security(verify_api_key)
):

    # --------------------------------------------------------
    # Count every prediction request
    # --------------------------------------------------------

    PREDICTION_REQUESTS.inc()

    try:

        # ----------------------------------------------------
        # Get client IP
        # ----------------------------------------------------

        client_ip = request.client.host
        print(f"DEBUG CLIENT IP: {client_ip}", flush=True)

        logger.info(
            "Prediction request received: IP=%s type=%s amount=%s",
            client_ip,
            transaction.type,
            transaction.amount
        )


        # ----------------------------------------------------
        # Rate limiting
        # ----------------------------------------------------

        if is_rate_limit_exceeded(
            client_ip
        ):

            RATE_LIMIT_EXCEEDED.inc()

            logger.warning(
                "Rate limit exceeded for IP: %s",
                client_ip
            )

            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Try again later."
            )


        # ----------------------------------------------------
        # Convert Pydantic object to dictionary
        # ----------------------------------------------------

        transaction_data = (
            transaction.model_dump()
        )


        # ----------------------------------------------------
        # Create Redis cache key
        # ----------------------------------------------------

        cache_key = create_cache_key(
            transaction_data
        )


        # ----------------------------------------------------
        # Check Redis cache
        # ----------------------------------------------------

        cached_prediction = (
            get_prediction_cache(
                cache_key
            )
        )


        # ----------------------------------------------------
        # Cache HIT
        # ----------------------------------------------------

        if cached_prediction is not None:

            CACHE_HITS.inc()

            PREDICTION_DECISIONS.labels(
                decision=cached_prediction[
                    "risk_decision"
                ]
            ).inc()

            logger.info(
                "Prediction returned from Redis cache"
            )

            return cached_prediction


        # ----------------------------------------------------
        # Cache MISS
        # ----------------------------------------------------

        CACHE_MISSES.inc()


        # ----------------------------------------------------
        # Convert transaction to DataFrame
        #
        # MLflow V3 expects raw transaction data.
        # V3 internally performs:
        #
        # raw data
        #     ↓
        # create_features()
        #     ↓
        # preprocessor
        #     ↓
        # XGBoost
        # ----------------------------------------------------

        transaction_df = pd.DataFrame(
            [transaction_data]
        )


        # ----------------------------------------------------
        # Fraud prediction using MLflow Champion
        # ----------------------------------------------------

        fraud_probability = fraud_model.predict(
            transaction_df
        )[0]


        fraud_probability = float(
            fraud_probability
        )


        # ----------------------------------------------------
        # Feature engineering for anomaly model
        # ----------------------------------------------------

        anomaly_input = create_features(
            transaction_df
        )


        # ----------------------------------------------------
        # Preprocess anomaly model input
        # ----------------------------------------------------

        anomaly_processed = (
            anomaly_preprocessor.transform(
                anomaly_input
            )
        )


        # ----------------------------------------------------
        # Calculate anomaly score
        # ----------------------------------------------------

        anomaly_score = (
            -anomaly_model.decision_function(
                anomaly_processed
            )[0]
        )


        anomaly_score = float(
            anomaly_score
        )


        # ----------------------------------------------------
        # Generate risk decision
        # ----------------------------------------------------

        risk_decision = get_risk_decision(
            fraud_probability
        )


        # ----------------------------------------------------
        # Create final prediction result
        # ----------------------------------------------------

        result = {
            "fraud_probability": fraud_probability,
            "risk_decision": risk_decision,
            "anomaly_score": anomaly_score
        }


        # ----------------------------------------------------
        # Save prediction to Redis
        # ----------------------------------------------------

        set_prediction_cache(
            cache_key,
            result,
            ttl=300
        )

        logger.info(
            "Prediction cached in Redis for 300 seconds"
        )


        # ----------------------------------------------------
        # Save prediction to PostgreSQL
        # ----------------------------------------------------

        save_prediction(
            transaction=transaction_data,
            prediction=result
        )

        logger.info(
            "Prediction saved to database"
        )


        # ----------------------------------------------------
        # Record prediction decision
        # ----------------------------------------------------

        PREDICTION_DECISIONS.labels(
            decision=result["risk_decision"]
        ).inc()


        # ----------------------------------------------------
        # Log prediction result
        # ----------------------------------------------------

        logger.info(
            "Prediction completed: decision=%s probability=%.6f",
            result["risk_decision"],
            result["fraud_probability"]
        )


        return result


    except HTTPException:

        raise


    except Exception:

        logger.exception(
            "Prediction failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Internal prediction error"
        )