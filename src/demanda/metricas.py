"""Métricas de error de pronóstico y de clasificación.

No hay MAPE a propósito: divide por la demanda de cada mes y no se puede
calcular cuando vale cero, que pasa seguido en los productos de poca venta.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Pronóstico
# --------------------------------------------------------------------------
def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    """Error absoluto medio, en unidades."""
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Σ|error| / Σ demanda. Pondera por volumen y no explota con ceros."""
    y, yhat = np.asarray(y), np.asarray(yhat)
    total = np.abs(y).sum()
    return float(np.abs(y - yhat).sum() / total) if total > 0 else float("nan")


def sesgo(y: np.ndarray, yhat: np.ndarray) -> float:
    """(Σ pronóstico − Σ demanda) / Σ demanda. Negativo: el modelo pide de menos."""
    y, yhat = np.asarray(y), np.asarray(yhat)
    total = y.sum()
    return float((yhat.sum() - total) / total) if total > 0 else float("nan")


def mase(y: np.ndarray, yhat: np.ndarray, escala: np.ndarray) -> float:
    """Error absoluto dividido por la escala de cada serie, promediado.

    Las filas con escala cero o nula (historia constante) quedan afuera.
    """
    y, yhat, escala = np.asarray(y), np.asarray(yhat), np.asarray(escala)
    ok = np.isfinite(escala) & (escala > 0)
    return float(np.mean(np.abs(y[ok] - yhat[ok]) / escala[ok])) if ok.any() else float("nan")


def escala_mase(Y_mensual: np.ndarray, hasta: int) -> np.ndarray:
    """MAE dentro de la historia del naive de un paso (Hyndman y Koehler, 2006).

    Se calcula con la demanda real completa hasta el mes `hasta`, también en los
    escenarios con menos historia, para que la vara sea la misma en todos.
    """
    dif = np.abs(np.diff(Y_mensual[:, : hasta + 1], axis=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(dif, axis=1)


def por_origen(pred: pd.DataFrame, claves: tuple[str, ...] = ("modelo",)) -> pd.DataFrame:
    """Métricas de cada modelo en cada mes pronosticado.

    `pred` tiene una fila por serie, modelo y mes, con `real`, `pronostico` y
    `escala`.
    """
    grupos = pred.groupby([*claves, "mes"], sort=False, observed=True)
    return grupos.apply(
        lambda g: pd.Series(
            {
                "mae": mae(g["real"], g["pronostico"]),
                "wape": wape(g["real"], g["pronostico"]),
                "sesgo": sesgo(g["real"], g["pronostico"]),
                "mase": mase(g["real"], g["pronostico"], g["escala"]),
                "aplica": g["aplica"].mean(),
                "series": len(g),
            }
        ),
        include_groups=False,
    ).reset_index()


def resumen(metricas: pd.DataFrame, claves: tuple[str, ...] = ("modelo",)) -> pd.DataFrame:
    """Promedio, desvío y peor origen de cada modelo, nunca un número suelto."""
    g = metricas.groupby(list(claves), sort=False, observed=True)
    out = g.agg(
        wape=("wape", "mean"),
        wape_desvio=("wape", "std"),
        wape_peor=("wape", "max"),
        sesgo=("sesgo", "mean"),
        mase=("mase", "mean"),
        mae=("mae", "mean"),
        aplica=("aplica", "mean"),
        origenes=("mes", "nunique"),
    )
    return out.reset_index()


def mejora(metricas: pd.DataFrame, modelo: str, referencia: str) -> dict:
    """Cuánto mejora `modelo` sobre `referencia` en WAPE, y en cuántos orígenes gana."""
    a = metricas[metricas["modelo"] == modelo].set_index("mes")["wape"]
    b = metricas[metricas["modelo"] == referencia].set_index("mes")["wape"]
    a, b = a.align(b, join="inner")
    return {
        "mejora": float(1 - a.mean() / b.mean()),
        "ganados": int((a < b).sum()),
        "origenes": int(len(a)),
    }


# --------------------------------------------------------------------------
# Clasificación (anexo)
# --------------------------------------------------------------------------
def clasificacion(y: np.ndarray, yhat: np.ndarray, positiva: int = 1) -> dict:
    """Precisión, recall y F1 de la clase `positiva`, más el accuracy."""
    y, yhat = np.asarray(y), np.asarray(yhat)
    vp = int(((yhat == positiva) & (y == positiva)).sum())
    fp = int(((yhat == positiva) & (y != positiva)).sum())
    fn = int(((yhat != positiva) & (y == positiva)).sum())
    precision = vp / (vp + fp) if vp + fp else 0.0
    recall = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": float((y == yhat).mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "alarmas": vp + fp,
        "aciertos": vp,
        "casos": vp + fn,
    }
