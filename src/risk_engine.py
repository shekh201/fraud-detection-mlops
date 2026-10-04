ALLOW_THRESHOLD = 0.01
BLOCK_THRESHOLD = 0.99


def get_risk_decision(fraud_probability):
    
    if fraud_probability < ALLOW_THRESHOLD:
        return "ALLOW"
    
    elif fraud_probability < BLOCK_THRESHOLD:
        return "REVIEW"
    
    else:
        return "BLOCK"