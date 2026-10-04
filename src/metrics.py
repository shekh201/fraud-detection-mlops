from prometheus_client import Counter, Gauge, Histogram


# ============================================================
# 1. Total Prediction Requests
# ============================================================

PREDICTION_REQUESTS = Counter(
    "fraud_prediction_requests_total",
    "Total number of prediction requests"
)


# ============================================================
# 2. Prediction Decisions
# ============================================================

PREDICTION_DECISIONS = Counter(
    "fraud_prediction_decisions_total",
    "Total number of fraud risk decisions",
    ["decision"]
)


# ============================================================
# 3. Prediction Latency
# ============================================================

PREDICTION_LATENCY = Histogram(
    "fraud_prediction_latency_seconds",
    "Time taken to process prediction requests"
)


# ============================================================
# 4. Authentication Failures
# ============================================================

AUTH_FAILURES = Counter(
    "fraud_auth_failures_total",
    "Total number of API authentication failures"
)


# ============================================================
# 5. Rate Limit Violations
# ============================================================

RATE_LIMIT_EXCEEDED = Counter(
    "fraud_rate_limit_exceeded_total",
    "Total number of rate limit violations"
)


# ============================================================
# 6. Redis Cache Hits
# ============================================================

CACHE_HITS = Counter(
    "fraud_cache_hits_total",
    "Total number of prediction cache hits"
)


# ============================================================
# 7. Redis Cache Misses
# ============================================================

CACHE_MISSES = Counter(
    "fraud_cache_misses_total",
    "Total number of prediction cache misses"
)


# ============================================================
# 8. Feature Drift - PSI
# ============================================================

FEATURE_DRIFT_PSI = Gauge(
    "fraud_feature_drift_psi",
    "PSI drift score for fraud detection features",
    ["feature"]
)


# ============================================================
# 9. Feature Drift - Status
# ============================================================

FEATURE_DRIFT_STATUS = Gauge(
    "fraud_feature_drift_status",
    "Feature drift status: 0=LOW, 1=MEDIUM, 2=HIGH",
    ["feature"]
)

# ============================================================
# 10. Drift Monitor Status
# ============================================================

DRIFT_MONITOR_STATUS = Gauge(
    "fraud_drift_monitor_status",
    "Drift monitor status: 0=SUCCESS, 1=INSUFFICIENT_DATA, 2=ERROR"
)