"""Implementación paralela de la simulación Monte Carlo."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src.simulation.model import NormalReturnModel
from src.simulation.sequential import SimulationConfig


def _simulate_chunk(
    scenarios: int,
    horizon: int,
    initial_value: float,
    mean: float,
    volatility: float,
    seed: int,
) -> np.ndarray:
    """Simula un bloque de escenarios."""

    rng = np.random.default_rng(seed)

    final_values = np.empty(
        scenarios,
        dtype=np.float64,
    )

    for scenario_index in range(
        scenarios
    ):

        returns = rng.normal(
            loc=mean,
            scale=volatility,
            size=horizon,
        )

        final_values[scenario_index] = (
            initial_value
            * np.exp(
                np.sum(returns)
            )
        )

    return final_values


def _build_chunks(
    total_scenarios: int,
    workers: int,
) -> list[int]:
    """Divide los escenarios entre workers."""

    base_size, remainder = divmod(
        total_scenarios,
        workers,
    )

    chunks = [
        base_size
        for _ in range(workers)
    ]

    for index in range(remainder):
        chunks[index] += 1

    return [
        chunk
        for chunk in chunks
        if chunk > 0
    ]


def simulate_parallel_final_values(
    model: NormalReturnModel,
    config: SimulationConfig,
    workers: int,
) -> np.ndarray:
    """Ejecuta escenarios utilizando múltiples procesos.

    Args:
        model: Modelo financiero.
        config: Configuración.
        workers: Número de procesos.

    Returns:
        Valores finales de todos los escenarios.
    """
    if workers <= 0:
        raise ValueError(
            "workers debe ser mayor que cero."
        )

    chunks = _build_chunks(
        config.scenarios,
        workers,
    )

    seed_sequence = np.random.SeedSequence(
        config.seed
    )

    child_sequences = (
        seed_sequence.spawn(
            len(chunks)
        )
    )

    child_seeds = [
        int(
            child.generate_state(
                1,
                dtype=np.uint64,
            )[0]
        )
        for child in child_sequences
    ]

    with ProcessPoolExecutor(
        max_workers=len(chunks)
    ) as executor:

        futures = [
            executor.submit(
                _simulate_chunk,
                chunk,
                config.horizon,
                config.initial_value,
                model.mean,
                model.volatility,
                seed,
            )
            for chunk, seed in zip(
                chunks,
                child_seeds,
            )
        ]

        results = [
            future.result()
            for future in futures
        ]

    return np.concatenate(
        results
    )