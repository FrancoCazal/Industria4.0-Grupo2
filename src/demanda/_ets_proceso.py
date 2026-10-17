"""Ajusta AutoETS en un proceso aparte del notebook.

Uso interno de `modelos.ETS`: `python -m demanda._ets_proceso entrada.parquet salida.parquet estacion h`.
Dentro de un kernel de Jupyter en Windows, el pool de procesos de statsforecast
a veces queda bloqueado para siempre; en un proceso de Python normal funciona.
"""

from __future__ import annotations

import sys
import warnings

import pandas as pd


def ajustar(largo: pd.DataFrame, estacion: int, h: int, n_jobs: int = -1) -> pd.DataFrame:
    """AutoETS sobre todas las series de `largo` (columnas unique_id, ds, y)."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS, Naive

    sf = StatsForecast(
        models=[AutoETS(season_length=estacion, alias="ets")],
        freq=1,
        n_jobs=n_jobs,
        fallback_model=Naive(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sf.forecast(df=largo, h=h)


def main() -> None:
    entrada, salida, estacion, h = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    ajustar(pd.read_parquet(entrada), estacion, h).to_parquet(salida, index=False)


if __name__ == "__main__":
    main()
