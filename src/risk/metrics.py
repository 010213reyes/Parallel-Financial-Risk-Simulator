"""Métricas financieras y de riesgo."""

from __future__ import annotations

import numpy as np


def calculate_mean_return(
    returns: np.ndarray,
) -> float:
    """Calcula el rendimiento promedio."""
    return float(np.mean(returns))


def calculate_volatility(
    returns: np.ndarray,
) -> float:
    """Calcula la desviación estándar de los rendimientos."""
    return float(np.std(returns))


def calculate_loss_probability(
    final_values: np.ndarray,
    initial_value: float,
) -> float:
    """Calcula la proporción de escenarios con pérdida."""
    if initial_value <= 0:
        raise ValueError(
            "initial_value debe ser mayor que cero."
        )

    losses = final_values < initial_value

    return float(np.mean(losses))


def calculate_percentiles(
    values: np.ndarray,
    percentiles: list[float],
) -> dict[float, float]:
    """Calcula percentiles de una distribución."""

    if len(values) == 0:
        raise ValueError(
            "values no puede estar vacío."
        )

    percentile_values = np.percentile(
        values,
        percentiles,
    )

    return {
        percentile: float(value)
        for percentile, value in zip(
            percentiles,
            percentile_values,
        )
    }


def calculate_var(
    returns: np.ndarray,
    confidence_level: float = 0.95,
) -> float:
    """Calcula Value at Risk histórico sobre rendimientos."""

    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level debe estar entre 0 y 1."
        )

    percentile = (
        1 - confidence_level
    ) * 100

    return float(
        -np.percentile(
            returns,
            percentile,
        )
    )


def calculate_expected_shortfall(
    returns: np.ndarray,
    confidence_level: float = 0.95,
) -> float:
    """Calcula Expected Shortfall histórico."""

    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level debe estar entre 0 y 1."
        )

    var_threshold = np.percentile(
        returns,
        (1 - confidence_level) * 100,
    )

    tail_losses = returns[
        returns <= var_threshold
    ]

    if len(tail_losses) == 0:
        return float(-var_threshold)

    return float(
        -np.mean(tail_losses)
    )