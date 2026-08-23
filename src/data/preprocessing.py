"""Funciones de limpieza y transformación de datos financieros."""

import numpy as np
import pandas as pd


def prepare_price_data(
    dataframe: pd.DataFrame,
    date_column: str = "observation_date",
    price_column: str = "SP500",
) -> pd.DataFrame:
    """
    Limpia y prepara una serie histórica de precios.

    Parameters
    ----------
    dataframe:
        Datos históricos originales.
    date_column:
        Nombre de la columna de fecha.
    price_column:
        Nombre de la columna del precio o índice.

    Returns
    -------
    pandas.DataFrame
        Datos ordenados y limpios.
    """
    required_columns = {date_column, price_column}

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(missing_columns)}"
        )

    result = dataframe[[date_column, price_column]].copy()

    result[date_column] = pd.to_datetime(
        result[date_column],
        errors="coerce",
    )

    result[price_column] = pd.to_numeric(
        result[price_column],
        errors="coerce",
    )

    result = result.dropna(
        subset=[date_column, price_column]
    )

    result = result.drop_duplicates(
        subset=[date_column]
    )

    result = result.sort_values(
        by=date_column
    )

    result = result.reset_index(drop=True)

    if (result[price_column] <= 0).any():
        raise ValueError(
            "La serie contiene precios o valores no positivos."
        )

    return result


def calculate_log_returns(
    dataframe: pd.DataFrame,
    price_column: str = "SP500",
) -> pd.DataFrame:
    """
    Calcula rendimientos logarítmicos diarios.

    Formula
    -------
    r_t = ln(P_t / P_{t-1})

    Parameters
    ----------
    dataframe:
        Datos financieros preparados.
    price_column:
        Columna de precios.

    Returns
    -------
    pandas.DataFrame
        DataFrame con la columna ``log_return``.
    """
    if price_column not in dataframe.columns:
        raise ValueError(
            f"No existe la columna de precios: {price_column}"
        )

    result = dataframe.copy()

    result["log_return"] = np.log(
        result[price_column]
        / result[price_column].shift(1)
    )

    result = result.dropna(
        subset=["log_return"]
    )

    return result.reset_index(drop=True)