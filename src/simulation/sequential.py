"""Implementación secuencial de simulaciones Monte Carlo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.simulation.model import NormalReturnModel


@dataclass(frozen=True)
class SimulationConfig:
    """Configuración de una simulación Monte Carlo.

    Attributes:
        scenarios: Número de escenarios.
        horizon: Número de periodos simulados.
        initial_value: Valor inicial de la inversión.
        seed: Semilla para reproducibilidad.
    """

    scenarios: int
    horizon: int
    initial_value: float
    seed: int | None = None

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


def simulate_sequential(
    model: NormalReturnModel,
    config: SimulationConfig,
) -> np.ndarray:
    """Ejecuta una simulación Monte Carlo secuencial.

    Cada escenario se genera independientemente de los demás.

    Args:
        model: Modelo estadístico de los rendimientos.
        config: Configuración de la simulación.

    Returns:
        Matriz con forma:

        (scenarios, horizon + 1)

        La primera columna contiene el valor inicial.
    """
    rng = np.random.default_rng(config.seed)

    simulations = np.empty(
        (
            config.scenarios,
            config.horizon + 1,
        ),
        dtype=np.float64,
    )

    simulations[:, 0] = config.initial_value

    for scenario_index in range(config.scenarios):

        returns = model.generate_returns(
            rng=rng,
            size=config.horizon,
        )

        cumulative_returns = np.cumsum(
            returns
        )

        simulations[scenario_index, 1:] = (
            config.initial_value
            * np.exp(cumulative_returns)
        )

    return simulations


def simulate_sequential_final_values(
    model: NormalReturnModel,
    config: SimulationConfig,
) -> np.ndarray:
    """Simula escenarios conservando únicamente el valor final.

    Esta función se utilizará principalmente para benchmarks,
    ya que evita almacenar todas las trayectorias.

    Args:
        model: Modelo estadístico de los rendimientos.
        config: Configuración de la simulación.

    Returns:
        Valor final de cada escenario.
    """
    rng = np.random.default_rng(config.seed)

    final_values = np.empty(
        config.scenarios,
        dtype=np.float64,
    )

    for scenario_index in range(
        config.scenarios
    ):
        returns = model.generate_returns(
            rng=rng,
            size=config.horizon,
        )

        final_values[scenario_index] = (
            config.initial_value
            * np.exp(
                np.sum(returns)
            )
        )

    return final_values