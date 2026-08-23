"""Modelos matemáticos utilizados por la simulación financiera."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NormalReturnModel:
    """Modelo de rendimientos basado en una distribución normal.

    Attributes:
        mean: Media del rendimiento diario.
        volatility: Desviación estándar del rendimiento diario.
    """

    mean: float
    volatility: float

    def generate_returns(
        self,
        rng: np.random.Generator,
        size: int,
    ) -> np.ndarray:
        """Genera rendimientos aleatorios.

        Args:
            rng: Generador de números aleatorios.
            size: Cantidad de rendimientos a generar.

        Returns:
            Arreglo NumPy con los rendimientos generados.
        """
        if size <= 0:
            raise ValueError(
                "size debe ser mayor que cero."
            )

        if self.volatility < 0:
            raise ValueError(
                "volatility no puede ser negativa."
            )

        return rng.normal(
            loc=self.mean,
            scale=self.volatility,
            size=size,
        )