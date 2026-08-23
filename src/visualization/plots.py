"""Funciones de visualización reutilizables."""

import matplotlib.pyplot as plt
import pandas as pd


def plot_price_series(
    dataframe: pd.DataFrame,
    date_column: str,
    price_column: str,
) -> None:
    """Grafica la evolución histórica del precio."""
    plt.figure(figsize=(12, 5))

    plt.plot(
        dataframe[date_column],
        dataframe[price_column],
    )

    plt.title("Evolución histórica")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_returns(
    dataframe: pd.DataFrame,
    return_column: str = "log_return",
) -> None:
    """Grafica la serie de rendimientos logarítmicos."""
    plt.figure(figsize=(12, 5))

    plt.plot(
        dataframe[return_column],
    )

    plt.title("Rendimientos logarítmicos")
    plt.xlabel("Observación")
    plt.ylabel("Rendimiento")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()