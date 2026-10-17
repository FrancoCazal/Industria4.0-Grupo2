"""Backtest de origen móvil y escenarios del experimento de madurez.

En cada origen los modelos solo ven lo que la empresa registró hasta ese mes.
La evaluación es siempre contra la demanda real completa (`verdad`), porque la
empresa vive la demanda real aunque haya registrado mal la suya.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import pandas as pd

from demanda import config as cfg
from demanda import features as F
from demanda import metricas as M
from demanda.datos import Panel
from demanda.modelos import Contexto, MediaMovil, Modelo


def correr(
    panel: Panel,
    verdad: np.ndarray,
    meses_objetivo: Sequence[pd.Period],
    modelos: Sequence[Modelo],
    *,
    historia_meses: int | None = None,
    k_respaldo: int = 6,
    etiqueta: str = "",
    verbose: bool = True,
) -> pd.DataFrame:
    """Corre todos los modelos en cada mes objetivo y devuelve una fila por serie.

    `panel` es lo que la empresa registró, en su frecuencia. `verdad` es la
    demanda mensual real, series × meses. Con `historia_meses` se borran los
    datos anteriores a esa ventana en cada origen, antes de calcular variables y
    entrenar.
    """
    p = panel.p
    pasos = F.pasos_objetivo(p)
    respaldo = MediaMovil(k_respaldo)

    # Sin recorte, las variables de un origen no dependen del origen de prueba:
    # se calculan una sola vez para todos los orígenes de entrenamiento.
    precalculado = None
    if historia_meses is None:
        ultimo = panel.fin_de_mes(max(meses_objetivo) - cfg.HORIZONTE) - max(pasos)
        precalculado = F.ejemplos(panel, F.origenes_fin_de_mes(panel, ultimo), pasos)

    filas = []
    for mes in meses_objetivo:
        t0 = time.perf_counter()
        mes_origen = mes - cfg.HORIZONTE
        o = panel.fin_de_mes(mes_origen)
        o_mes = panel.indice_mes(mes_origen)
        vista, desde = panel, 0
        if historia_meses is not None:
            desde = o - historia_meses * p + 1
            Y = panel.Y.copy()
            Y[:, : max(desde, 0)] = np.nan
            vista = panel.con_Y(Y)
        ctx = Contexto(
            panel=vista,
            origen=o,
            pasos=pasos,
            origenes_entrenamiento=F.origenes_fin_de_mes(vista, o - max(pasos), max(desde, 0)),
            precalculado=precalculado,
        )

        # Se evalúan las series que ya existían en el origen, según la verdad.
        evaluar = ~np.isnan(verdad[:, : o_mes + 1]).all(axis=1)
        real = verdad[:, panel.indice_mes(mes)]
        escala = M.escala_mase(verdad, o_mes)
        base = np.nan_to_num(respaldo.pronosticar(ctx), nan=0.0)

        for modelo in modelos:
            t1 = time.perf_counter()
            pred = np.asarray(modelo.pronosticar(ctx), dtype="float64")
            aplica = ~np.isnan(pred)
            pred = np.clip(np.where(aplica, pred, base), 0, None)
            filas.append(
                pd.DataFrame(
                    {
                        "fila": np.flatnonzero(evaluar),
                        "modelo": modelo.nombre,
                        "familia": modelo.familia,
                        "mes": mes,
                        "real": real[evaluar],
                        "pronostico": pred[evaluar],
                        "aplica": aplica[evaluar],
                        "escala": escala[evaluar],
                        "segundos": time.perf_counter() - t1,
                    }
                )
            )
        if verbose:
            print(f"{etiqueta} {mes} listo en {time.perf_counter() - t0:.1f} s")

    out = pd.concat(filas, ignore_index=True)
    out["serie"] = panel.series["serie"].to_numpy()[out["fila"]]
    return out


def evaluar(pred: pd.DataFrame, claves: tuple[str, ...] = ("modelo",)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Métricas por origen y su resumen con dispersión."""
    por_origen = M.por_origen(pred, claves)
    return por_origen, M.resumen(por_origen, claves)


#: Un modelo que no se pudo aplicar a la mitad de las series no se reporta en
#: ese escenario: sus cifras serían en realidad las de la media móvil de respaldo.
MIN_APLICA = 0.5


def guardar_pred(pred: pd.DataFrame, archivo) -> None:
    """Guarda predicciones en la caché, con el mes como texto."""
    archivo.parent.mkdir(parents=True, exist_ok=True)
    pred.assign(mes=pred["mes"].astype(str)).to_parquet(archivo, index=False)


def leer_pred(archivo) -> pd.DataFrame:
    """Lee predicciones de la caché, con el mes como período mensual."""
    pred = pd.read_parquet(archivo)
    pred["mes"] = pd.PeriodIndex(pred["mes"], freq="M")
    return pred


def con_cache(archivo, recalcular: bool, calcular) -> pd.DataFrame:
    """Devuelve la caché si existe y no se pide recalcular; si no, calcula y guarda."""
    if archivo.exists() and not recalcular:
        print(f"Leído de la caché: {archivo.name}")
        return leer_pred(archivo)
    pred = calcular()
    guardar_pred(pred, archivo)
    return pred


def vale_la_pena(metricas: pd.DataFrame, referencias: Sequence[str], ml: Sequence[str]) -> dict:
    """Aplica el criterio declarado en la guía (sección 9) a un escenario.

    El ML vale la pena si su mejor modelo baja el WAPE al menos
    `UMBRAL_MEJORA_ML` frente a la mejor referencia simple y le gana en al menos
    `ORIGENES_GANADOS_MIN` orígenes. Las referencias que no se pudieron aplicar
    en el escenario no compiten.
    """
    g = metricas.groupby("modelo")
    wape, aplica = g["wape"].mean(), g["aplica"].mean()
    refs = [r for r in referencias if r in wape.index and aplica[r] >= MIN_APLICA]
    mls = [m for m in ml if m in wape.index]
    mejor_ref = wape[refs].idxmin()
    mejor_ml = wape[mls].idxmin()
    r = M.mejora(metricas, mejor_ml, mejor_ref)
    r.update(
        mejor_ml=mejor_ml,
        mejor_referencia=mejor_ref,
        wape_ml=float(wape[mejor_ml]),
        wape_referencia=float(wape[mejor_ref]),
    )
    r["vale_la_pena"] = bool(r["mejora"] >= cfg.UMBRAL_MEJORA_ML and r["ganados"] >= cfg.ORIGENES_GANADOS_MIN)
    return r
