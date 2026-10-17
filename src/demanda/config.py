"""Rutas, semilla y calendario del backtest. Un solo lugar para cada constante."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[2]
DATA_RAW = RAIZ / "data" / "raw"
DATA_PROCESSED = RAIZ / "data" / "processed"
RESULTADOS = RAIZ / "resultados"
FIGURAS = RESULTADOS / "figuras"
TABLAS = RESULTADOS / "tablas"
POWERBI = RESULTADOS / "powerbi"

KAGGLE_DATASET = "felixzhao/productdemandforecasting"
ARCHIVO_CRUDO = "Historical Product Demand.csv"

# Salidas de 01, entradas de 02 y 03
ARCHIVO_LIMPIO = DATA_PROCESSED / "demanda_limpia.parquet"
ARCHIVO_PANEL = DATA_PROCESSED / "panel_mensual.parquet"
ARCHIVO_DIM_SERIE = DATA_PROCESSED / "dim_serie.parquet"

SEMILLA = 42

# --------------------------------------------------------------------------
# Calendario
# --------------------------------------------------------------------------
#: Primer y último mes completos del dataset. 2011 y 2017 quedan afuera.
MES_INICIO = pd.Period("2012-01", freq="M")
MES_FIN = pd.Period("2016-12", freq="M")

#: Con datos hasta el cierre del mes t se pronostica el mes t + HORIZONTE.
HORIZONTE = 2

#: Meses pronosticados en cada etapa. El origen de cada uno es mes - HORIZONTE.
MESES_SELECCION = list(pd.period_range("2015-02", "2015-12", freq="M"))[::2]
MESES_PRUEBA = list(pd.period_range("2016-01", "2016-12", freq="M"))
MESES_EXPERIMENTO = list(pd.period_range("2016-02", "2016-12", freq="M"))[::2]

#: Mes del pronóstico "real" que se entrega a Power BI.
MES_FUTURO = MES_FIN + HORIZONTE

# --------------------------------------------------------------------------
# Experimento de madurez
# --------------------------------------------------------------------------
HISTORIAS_MESES = [6, 12, 24, 36]
FALTANTES = [0.05, 0.15, 0.30]

#: Criterio declarado antes de correr el experimento (guía, sección 9).
UMBRAL_MEJORA_ML = 0.05
ORIGENES_GANADOS_MIN = 4


def asegurar_carpetas() -> None:
    """Crea las carpetas de salida si no existen."""
    for carpeta in (DATA_RAW, DATA_PROCESSED, FIGURAS, TABLAS, POWERBI):
        carpeta.mkdir(parents=True, exist_ok=True)
