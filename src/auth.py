import os
import secrets

from dotenv import load_dotenv
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from src.metrics import AUTH_FAILURES


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# Load API key
# --------------------------------------------------

API_KEY = os.getenv("API_KEY")


# --------------------------------------------------
# Define API key header
# --------------------------------------------------

API_KEY_HEADER = APIKeyHeader(
    name="X-API-Key",
    auto_error=False
)


# --------------------------------------------------
# Verify API key
# --------------------------------------------------

def verify_api_key(
    api_key: str | None = Security(API_KEY_HEADER)
):

    # ----------------------------------------------
    # Check if API key was provided
    # ----------------------------------------------

    if api_key is None:

        AUTH_FAILURES.inc()

        raise HTTPException(
            status_code=401,
            detail="API key is missing"
        )


    # ----------------------------------------------
    # Check server configuration
    # ----------------------------------------------

    if API_KEY is None:

        raise HTTPException(
            status_code=500,
            detail="API key is not configured"
        )


    # ----------------------------------------------
    # Secure API key comparison
    # ----------------------------------------------

    if not secrets.compare_digest(
        api_key,
        API_KEY
    ):

        AUTH_FAILURES.inc()

        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )


    # ----------------------------------------------
    # Authentication successful
    # ----------------------------------------------

    return api_key