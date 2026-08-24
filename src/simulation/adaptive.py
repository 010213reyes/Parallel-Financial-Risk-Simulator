"""Simulación Monte Carlo con criterio adaptativo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AdaptiveConfig:
    """Configuración del simulador adaptativo."""

    initial_scenarios: int
    batch_size: int
    max_scenarios: int
    target_standard_error: float
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        """Valida la configuración."""

        if self.initial_scenarios <= 0:
            raise ValueError(
                "initial_scenarios debe ser mayor que cero."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size debe ser mayor que cero."
            )

        if self.max_scenarios < self.initial_scenarios:
            raise ValueError(
                "max_scenarios debe ser >= initial_scenarios."
            )

        if self.target_standard_error <= 0:
            raise ValueError(
                "target_standard_error debe ser mayor que cero."
            )


def calculate_standard_error(
    values: np.ndarray,
) -> float:
    """Calcula el error estándar de la media."""

    if len(values) < 2:
        return float("inf")

    return float(
        np.std(values, ddof=1)
        / np.sqrt(len(values))
    )


def should_continue(
    standard_error: float,
    target_standard_error: float,
    number_of_scenarios: int,
    max_scenarios: int,
) -> bool:
    """Determina si deben generarse más escenarios."""

    return (
        standard_error > target_standard_error
        and number_of_scenarios < max_scenarios
    )
    