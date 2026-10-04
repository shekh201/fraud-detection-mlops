import numpy as np

from src.drift import calculate_psi, interpret_psi


def test_same_distribution_has_low_psi():
    reference = np.array([
        10, 20, 30, 40, 50,
        60, 70, 80, 90, 100
    ])

    current = np.array([
        10, 20, 30, 40, 50,
        60, 70, 80, 90, 100
    ])

    psi = calculate_psi(
        reference,
        current
    )

    assert psi < 0.10


def test_different_distribution_has_high_psi():
    reference = np.array([
        10, 20, 30, 40, 50,
        60, 70, 80, 90, 100
    ])

    current = np.array([
        1000, 1100, 1200, 1300, 1400,
        1500, 1600, 1700, 1800, 1900
    ])

    psi = calculate_psi(
        reference,
        current
    )

    assert psi >= 0.25


def test_psi_interpretation():
    assert interpret_psi(0.05) == "LOW"
    assert interpret_psi(0.15) == "MEDIUM"
    assert interpret_psi(0.30) == "HIGH"