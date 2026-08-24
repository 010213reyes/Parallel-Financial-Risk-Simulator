"""Herramientas para medir rendimiento."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable


@dataclass(frozen=True)
class BenchmarkResult:
    """Resultado de un benchmark."""

    scenarios: int
    elapsed_seconds: float
    scenarios_per_second: float


def benchmark_simulation(
    simulation_function: Callable[[], object],
    scenarios: int,
) -> BenchmarkResult:
    """Mide el tiempo de ejecución de una simulación.

    Args:
        simulation_function: Función a ejecutar.
        scenarios: Número de escenarios.

    Returns:
        Resultado del benchmark.
    """
    start_time = perf_counter()

    simulation_function()

    elapsed_seconds = (
        perf_counter() - start_time
    )

    scenarios_per_second = (
        scenarios / elapsed_seconds
    )

    return BenchmarkResult(
        scenarios=scenarios,
        elapsed_seconds=elapsed_seconds,
        scenarios_per_second=scenarios_per_second,
    )