"""Funciones para cargar datos financieros desde archivos locales."""

from pathlib import Path

import pandas as pd


def load_csv(file_path: str | Path) -> pd.DataFrame:
    """
    Carga un archivo CSV y devuelve un DataFrame.

    Parameters
    ----------
    file_path:
        Ruta al archivo CSV.

    Returns
    -------
    pandas.DataFrame
        Datos cargados.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.
    ValueError
        Si el archivo no tiene registros.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos: {path}"
        )

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"El archivo está vacío: {path}")

    return dataframe