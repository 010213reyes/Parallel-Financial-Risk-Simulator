"""Simulación Monte Carlo secuencial."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.simulation.model import NormalReturnModel


@dataclass(frozen=True)
class SimulationConfig:
    """Configuración de una simulación Monte Carlo.

    Attributes:
        scenarios: Número de escenarios.
        horizon: Número de períodos por escenario.
        initial_value: Valor inicial de la inversión.
        seed: Semilla para reproducibilidad.
    """

    scenarios: int
    horizon: int
    initial_value: float
    seed: int = 42

    def __post_init__(self) -> None:
        """Valida los parámetros de configuración."""
        if self.scenarios <= 0:
            raise ValueError(
                "scenarios debe ser mayor que cero."
            )

        if self.horizon <= 0:
            raise ValueError(
                "horizon debe ser mayor que cero."
            )

        if self.initial_value <= 0:
            raise ValueError(
                "initial_value debe ser mayor que cero."
            )


@dataclass(frozen=True)
class SimulationResult:
    """Resultados producidos por una simulación.

    Attributes:
        returns: Rendimientos simulados.
        prices: Trayectorias de precios.
    """

    returns: np.ndarray
    prices: np.ndarray


def simulate_sequential(
    model: NormalReturnModel,
    config: SimulationConfig,
) -> SimulationResult:
    """Ejecuta una simulación Monte Carlo secuencial.

    Cada escenario se genera de manera independiente utilizando
    un único proceso de ejecución.

    Args:
        model: Modelo utilizado para generar rendimientos.
        config: Configuración de la simulación.

    Returns:
        Resultados de la simulación.
    """
    rng = np.random.default_rng(config.seed)

    returns = np.empty(
        (
            config.scenarios,
            config.horizon,
        ),
        dtype=np.float64,
    )

    prices = np.empty_like(returns)

    for scenario_index in range(config.scenarios):
        scenario_returns = model.generate_returns(
            rng=rng,
            size=config.horizon,
        )

        scenario_prices = (
            config.initial_value
            * np.exp(
                np.cumsum(scenario_returns)
            )
        )

        returns[scenario_index] = scenario_returns
        prices[scenario_index] = scenario_prices

    return SimulationResult(
        returns=returns,
        prices=prices,
    )