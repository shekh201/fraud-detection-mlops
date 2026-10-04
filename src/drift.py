import numpy as np
import pandas as pd


def calculate_psi(reference, current, bins=10):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    breakpoints = np.percentile(
        reference,
        np.linspace(0, 100, bins + 1)
    )

    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 3:
        return 0.0

    reference_counts, _ = np.histogram(
        reference,
        bins=breakpoints
    )

    current_counts, _ = np.histogram(
        current,
        bins=breakpoints
    )

    reference_percent = reference_counts / len(reference)
    current_percent = current_counts / len(current)

    epsilon = 1e-6

    reference_percent = np.clip(
        reference_percent,
        epsilon,
        None
    )

    current_percent = np.clip(
        current_percent,
        epsilon,
        None
    )

    psi = np.sum(
        (current_percent - reference_percent)
        *
        np.log(current_percent / reference_percent)
    )

    return float(psi)


def interpret_psi(psi):

    if psi < 0.10:
        return "LOW"

    if psi < 0.25:
        return "MEDIUM"

    return "HIGH"

def calculate_categorical_psi(reference, current):
    reference = pd.Series(reference)
    current = pd.Series(current)

    categories = sorted(
        set(reference.dropna().unique())
        |
        set(current.dropna().unique())
    )

    reference_counts = (
        reference
        .value_counts(normalize=True)
        .reindex(categories, fill_value=0)
    )

    current_counts = (
        current
        .value_counts(normalize=True)
        .reindex(categories, fill_value=0)
    )

    epsilon = 1e-6

    reference_percent = np.clip(
        reference_counts.values,
        epsilon,
        None
    )

    current_percent = np.clip(
        current_counts.values,
        epsilon,
        None
    )

    psi = np.sum(
        (current_percent - reference_percent)
        *
        np.log(current_percent / reference_percent)
    )

    return float(psi)