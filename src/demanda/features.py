"""Variables (features) ancladas en el origen.

Regla antifuga: todo lo que entra a `X` se calcula con `Y[:, : origen + 1]`. El
único dato posterior al origen que se lee es el objetivo `y`, y solo cuando se
arma el entrenamiento.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from demanda import config as cfg
from demanda.datos import Panel

#: Cantidad de rezagos según la frecuencia: 12 meses o 8 semanas.
N_REZAGOS = {1: 12, 4: 8}
VENTANAS_MESES = (3, 6, 12)
CATEGORICAS = ["deposito", "categoria"]


def pasos_objetivo(p: int, horizonte: int = cfg.HORIZONTE) -> list[int]:
    """Pasos, en períodos desde el origen, que forman el mes objetivo.

    Mensual: `[2]`. Semanal: `[5, 6, 7, 8]`, las cuatro semanas del mes t + 2.
    """
    return list(range((horizonte - 1) * p + 1, horizonte * p + 1))


def _media(a: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=1)


def _desvio(a: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(a, axis=1)


def _mediana(a: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a, axis=1)


def tasa_croston(H: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Tasa por período de Croston SBA al final de la historia `H` (series × períodos).

    Suaviza por separado el tamaño de los pedidos y el intervalo entre ellos; la
    tasa es tamaño / intervalo, corregida por (1 - α/2) (Syntetos y Boylan). NaN
    si la serie no tiene ningún pedido en `H`.
    """
    n, T = H.shape
    tamano = np.full(n, np.nan)
    intervalo = np.full(n, np.nan)
    desde = np.zeros(n)
    for t in range(T):
        x = H[:, t]
        obs = ~np.isnan(x)
        desde[obs] += 1
        pedido = obs & (x > 0)
        primero = pedido & np.isnan(tamano)
        siguiente = pedido & ~primero
        tamano[primero] = x[primero]
        intervalo[primero] = desde[primero]
        tamano[siguiente] += alpha * (x[siguiente] - tamano[siguiente])
        intervalo[siguiente] += alpha * (desde[siguiente] - intervalo[siguiente])
        desde[pedido] = 0
    return (1 - alpha / 2) * tamano / np.maximum(intervalo, 1)


def nivel(Y: np.ndarray, origen: int, p: int) -> np.ndarray:
    """Nivel `s` de cada serie en el origen: promedio de los últimos 12 meses.

    Si es cero o no hay datos, el promedio de toda la historia; si también es
    cero, 1. Divide las cantidades para que el modelo aprenda patrones y no
    tamaños.
    """
    historia = Y[:, : origen + 1]
    s = _media(historia[:, max(0, origen - 12 * p + 1) :])
    todo = _media(historia)
    s = np.where(np.isnan(s) | (s <= 0), todo, s)
    return np.where(np.isnan(s) | (s <= 0), 1.0, s)


def variables_origen(panel: Panel, origen: int, pasos: list[int]) -> tuple[pd.DataFrame, np.ndarray]:
    """Variables de todas las series en un origen, una fila por serie y paso.

    Devuelve `X` y el nivel `s` de cada fila. Las filas siguen el orden
    `paso`, y dentro de cada paso, el orden de `panel.series`.
    """
    Y, p = panel.Y, panel.p
    if max(pasos) >= 12 * p:
        raise ValueError("el rezago anual tiene que caer antes del origen")
    historia = Y[:, : origen + 1]
    s = nivel(Y, origen, p)
    n = Y.shape[0]
    nan = np.full(n, np.nan)

    base: dict[str, np.ndarray] = {}
    for j in range(N_REZAGOS[p]):
        base[f"rez_{j}"] = (Y[:, origen - j] if origen - j >= 0 else nan) / s
    for w in VENTANAS_MESES:
        base[f"media_{w}m"] = _media(historia[:, max(0, origen - w * p + 1) :]) / s
    # Resúmenes robustos: la mediana ignora un pedido gigante suelto, y la tasa
    # de Croston le da al modelo la mejor referencia simple como variable.
    for w in (6, 12):
        base[f"mediana_{w}m"] = _mediana(historia[:, max(0, origen - w * p + 1) :]) / s
    base["tasa_croston"] = tasa_croston(historia) * p / s
    ult12 = historia[:, max(0, origen - 12 * p + 1) :]
    base["desvio_12m"] = _desvio(ult12) / s
    observados = (~np.isnan(ult12)).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        base["frac_ceros_12m"] = np.where(observados > 0, (ult12 == 0).sum(axis=1) / observados, np.nan)
    base["nivel_log"] = np.log(s)
    iniciada = ~np.isnan(historia).all(axis=1)
    primero = np.where(iniciada, (~np.isnan(historia)).argmax(axis=1), origen + 1)
    base["antiguedad_meses"] = (origen - primero + 1) / p

    bloques = []
    for h in pasos:
        b = pd.DataFrame(base)
        for anios, nombre in ((1, "mismo_periodo_anio_ant"), (2, "hace_dos_anios")):
            j = origen + h - anios * 12 * p
            b[nombre] = (Y[:, j] if j >= 0 else nan) / s
        b["mes_objetivo"] = int(panel.mes_de_periodo(origen + h))
        if p > 1:
            b["semana_objetivo"] = (origen + h) % p + 1
            b["paso"] = h
        for c in CATEGORICAS:
            b[c] = panel.series[c].to_numpy()
        bloques.append(b)
    X = pd.concat(bloques, ignore_index=True)
    return X, np.tile(s, len(pasos))


def ejemplos(
    panel: Panel, origenes: list[int], pasos: list[int], *, con_objetivo: bool = True
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Arma `X`, el objetivo escalado `y` y la metadata de cada fila.

    Solo entran series que ya existen en el origen. Con `con_objetivo=True`
    también se descartan las filas cuyo objetivo no se conoce.
    """
    Y = panel.Y
    n = Y.shape[0]
    Xs, ys, metas = [], [], []
    for o in origenes:
        X, s = variables_origen(panel, o, pasos)
        fila = np.tile(np.arange(n), len(pasos))
        paso = np.repeat(pasos, n)
        destino = o + paso
        y = np.full(len(fila), np.nan)
        dentro = destino < Y.shape[1]
        y[dentro] = Y[fila[dentro], destino[dentro]] / s[dentro]
        iniciada = ~np.isnan(Y[:, : o + 1]).all(axis=1)
        ok = iniciada[fila]
        if con_objetivo:
            ok &= ~np.isnan(y)
        Xs.append(X[ok])
        ys.append(y[ok])
        metas.append(pd.DataFrame({"fila": fila[ok], "origen": o, "paso": paso[ok], "s": s[ok]}))
    X = pd.concat(Xs, ignore_index=True)
    for c in CATEGORICAS:
        X[c] = pd.Categorical(X[c], categories=sorted(panel.series[c].unique()))
    return X, np.concatenate(ys), pd.concat(metas, ignore_index=True)


def origenes_fin_de_mes(panel: Panel, hasta: int, desde: int = 0) -> list[int]:
    """Orígenes de entrenamiento: el último período de cada mes en [desde, hasta]."""
    fines = np.arange(panel.p - 1, panel.Y.shape[1], panel.p)
    return [int(o) for o in fines if desde <= o <= hasta]
