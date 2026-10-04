from src.risk_engine import get_risk_decision


def test_very_low_risk_is_allowed():
    result = get_risk_decision(0.001)
    assert result == "ALLOW"


def test_medium_risk_goes_to_review():
    result = get_risk_decision(0.50)
    assert result == "REVIEW"


def test_very_high_risk_is_blocked():
    result = get_risk_decision(0.999)
    assert result == "BLOCK"