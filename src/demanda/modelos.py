"""Modelos de pronóstico: referencias, ETS, Ridge, LightGBM y la logística del anexo.

Todos responden lo mismo: dado un origen, la demanda de cada serie en el mes
objetivo (t + 2). Devuelven NaN donde no se pueden aplicar; el backtest completa
esas series con la media móvil y registra en qué fracción pasó.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from demanda import config as cfg
from demanda import features as F
from demanda.datos import Panel

FAMILIAS = {"referencia": "Referencia simple", "estadistico": "Estadístico", "ml": "Machine learning"}


# --------------------------------------------------------------------------
# Contexto de un origen
# --------------------------------------------------------------------------
@dataclass
class Contexto:
    """Lo que un modelo puede ver en un origen.

    `panel` es lo que la empresa registró: puede estar recortado (menos historia)
    o degradado (registros perdidos). Nada de lo que hay acá permite leer la
    demanda real del mes objetivo.
    """

    panel: Panel
    origen: int
    pasos: list[int]
    origenes_entrenamiento: list[int]
    precalculado: tuple[pd.DataFrame, np.ndarray, pd.DataFrame] | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def historia(self) -> np.ndarray:
        return self.panel.Y[:, : self.origen + 1]

    @property
    def n(self) -> int:
        return self.panel.Y.shape[0]

    def entrenamiento(self) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        if "ent" not in self._cache:
            if self.precalculado is not None:
                X, y, meta = self.precalculado
                ok = np.isin(meta["origen"].to_numpy(), self.origenes_entrenamiento)
                self._cache["ent"] = (X[ok], y[ok], meta[ok])
            else:
                self._cache["ent"] = F.ejemplos(self.panel, self.origenes_entrenamiento, self.pasos)
        return self._cache["ent"]

    def prueba(self) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        if "pru" not in self._cache:
            self._cache["pru"] = F.ejemplos(self.panel, [self.origen], self.pasos, con_objetivo=False)
        return self._cache["pru"]

    def agregar(self, pred_escalada: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
        """Desescala y suma los pasos de cada serie: pronóstico del mes objetivo."""
        out = np.full(self.n, np.nan)
        suma = pd.Series(np.clip(pred_escalada, 0, None) * meta["s"].to_numpy()).groupby(
            meta["fila"].to_numpy()
        ).sum()
        out[suma.index.to_numpy()] = suma.to_numpy()
        return out


class Modelo:
    """Interfaz común. `pronosticar` devuelve un valor por serie (NaN = no aplica)."""

    nombre = "modelo"
    familia = "referencia"

    def pronosticar(self, ctx: Contexto) -> np.ndarray:
        raise NotImplementedError


# --------------------------------------------------------------------------
# Referencias simples
# --------------------------------------------------------------------------
class Naive(Modelo):
    """El mes objetivo se vende lo mismo que el último período conocido."""

    nombre = "Naive"

    def pronosticar(self, ctx: Contexto) -> np.ndarray:
        return ctx.historia[:, -1] * len(ctx.pasos)


class NaiveEstacional(Modelo):
    """El mes objetivo se vende lo mismo que ese mes el año anterior."""

    nombre = "Naive estacional"

    def pronosticar(self, ctx: Contexto) -> np.ndarray:
        p = ctx.panel.p
        cols = [ctx.origen + h - 12 * p for h in ctx.pasos]
        if min(cols) < 0:
            return np.full(ctx.n, np.nan)
        # NaN si algún período del año anterior no existe para esa serie.
        return ctx.panel.Y[:, cols].sum(axis=1)


class MediaMovil(Modelo):
    """Promedio de los últimos `k` meses, la regla típica de una pyme."""

    def __init__(self, k: int = 6) -> None:
        self.k = k
        self.nombre = f"Media móvil {k}m"

    def pronosticar(self, ctx: Contexto) -> np.ndarray:
        p = ctx.panel.p
        ventana = ctx.historia[:, -self.k * p :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(ventana, axis=1) * len(ctx.pasos)


class CrostonSBA(Modelo):
    """Croston con la corrección de Syntetos y Boylan, para demanda intermitente.

    Suaviza por separado el tamaño de los pedidos y el intervalo entre ellos; la
    tasa por período es tamaño / intervalo, multiplicada por (1 - α/2).
    """

    nombre = "Croston SBA"

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha

    def pronosticar(self, ctx: Contexto) -> np.ndarray:
        H = ctx.historia
        tasa = F.tasa_croston(H, self.alpha)
        iniciada = ~np.isnan(H).all(axis=1)
        tasa = np.where(iniciada & np.isnan(tasa), 0.0, tasa)
        return tasa * len(ctx.pasos)


# --------------------------------------------------------------------------
# Estadístico
# --------------------------------------------------------------------------
#: Tiempo máximo de un ajuste de ETS antes de cortar con un error claro.
ETS_TIMEOUT_SEGUNDOS = 900


def _ets_paralelo(largo: pd.DataFrame, estacion: int, h: int) -> pd.DataFrame:
    """Ajusta AutoETS en paralelo, en un proceso de Python aparte.

    Dentro de un kernel de Jupyter en Windows, el pool de procesos de
    statsforecast a veces queda bloqueado para siempre; en un proceso normal
    funciona. Por eso el ajuste corre en `demanda._ets_proceso` y los datos van y
    vuelven en parquet. El resultado es idéntico al de llamarlo directo.
    """
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as carpeta:
        entrada, salida = Path(carpeta) / "entrada.parquet", Path(carpeta) / "salida.parquet"
        largo.to_parquet(entrada, index=False)
        src = str(Path(__file__).resolve().parents[1])
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")]))}
        r = subprocess.run(
            [sys.executable, "-m", "demanda._ets_proceso", str(entrada), str(salida), str(estacion), str(h)],
            env=env,
            capture_output=True,
            text=True,
            timeout=ETS_TIMEOUT_SEGUNDOS,
        )
        if r.returncode != 0:
            raise RuntimeError(f"ETS falló en el proceso aparte:\n{r.stderr[-2000:]}")
        return pd.read_parquet(salida)


class ETS(Modelo):
    """Suavizado exponencial con selección automática (statsforecast `AutoETS`).

    Un modelo por serie. Mensual: estacional (12) con 24 meses o más de historia
    y sin estacionalidad con menos. No se aplica con menos de 12 períodos. En
    semanal se usa sin estacionalidad: la estación anual tendría 48 períodos.
    """

    nombre = "ETS"
    familia = "estadistico"
    MIN_PERIODOS = 12

    def pronosticar(self, ctx: Contexto) -> np.ndarray:
        H = ctx.historia
        p = ctx.panel.p
        n_obs = (~np.isnan(H)).sum(axis=1)
        out = np.full(ctx.n, np.nan)
        grupos = {1: n_obs >= self.MIN_PERIODOS}
        if p == 1:
            grupos = {12: n_obs >= 24, 1: (n_obs >= self.MIN_PERIODOS) & (n_obs < 24)}
        for estacion, mascara in grupos.items():
            filas = np.flatnonzero(mascara)
            if filas.size == 0:
                continue
            sub = H[filas]
            i, j = np.nonzero(~np.isnan(sub))
            largo = pd.DataFrame({"unique_id": filas[i], "ds": j, "y": sub[i, j]})
            pron = _ets_paralelo(largo, estacion, max(ctx.pasos))
            pron["paso"] = pron["ds"] - ctx.origen
            pron = pron[pron["paso"].isin(ctx.pasos)]
            suma = pron.groupby("unique_id")["ets"].sum().clip(lower=0)
            out[suma.index.to_numpy().astype(int)] = suma.to_numpy()
        return out


# --------------------------------------------------------------------------
# Machine learning global
# --------------------------------------------------------------------------
#: Tope del objetivo escalado al entrenar: 12 veces el nivel de la serie. Un
#: salto mayor no se puede anticipar con estas variables y, sin tope, unas pocas
#: series nuevas con nivel minúsculo dominarían la pérdida cuadrática.
TOPE_OBJETIVO = 12.0


class ModeloGlobal(Modelo):
    """Un solo modelo para todas las series, sobre variables escaladas por nivel."""

    familia = "ml"

    def __init__(self) -> None:
        self.ultimo_ = None
        self.columnas_ = None
        self.segundos_ajuste_ = 0.0

    def _estimador(self, X: pd.DataFrame):
        raise NotImplementedError

    def _pesos(self, meta: pd.DataFrame) -> np.ndarray:
        """Peso proporcional al nivel de la serie, normalizado a promedio 1.

        La proporción es lo que alinea la pérdida con el WAPE. La normalización
        importa por la regularización: con pesos de millones de unidades, la
        penalización de Ridge (alpha) y la de LightGBM (reg_lambda) quedan
        insignificantes al lado del error ponderado y el modelo se ajusta sin
        regularizar.
        """
        s = meta["s"].to_numpy()
        return s / s.mean()

    def ajustar(self, X: pd.DataFrame, y: np.ndarray, meta: pd.DataFrame):
        t0 = time.perf_counter()
        est = self._estimador(X)
        est.fit(X, np.minimum(y, TOPE_OBJETIVO), **self._kwargs_peso(meta))
        self.ultimo_, self.columnas_ = est, list(X.columns)
        self.segundos_ajuste_ = time.perf_counter() - t0
        return est

    def _kwargs_peso(self, meta: pd.DataFrame) -> dict:
        return {"sample_weight": self._pesos(meta)}

    def pronosticar(self, ctx: Contexto) -> np.ndarray:
        X, y, meta = ctx.entrenamiento()
        est = self.ajustar(X, y, meta)
        Xp, _, metap = ctx.prueba()
        return ctx.agregar(est.predict(Xp), metap)


class LightGBM(ModeloGlobal):
    """Árboles de decisión con boosting (Ke et al., 2017)."""

    def __init__(self, objetivo: str = "l1", alpha: float | None = None, nombre: str | None = None) -> None:
        super().__init__()
        self.objetivo = objetivo
        self.alpha = alpha
        self.nombre = nombre or "LightGBM"

    def _estimador(self, X: pd.DataFrame):
        from lightgbm import LGBMRegressor

        extra = {"alpha": self.alpha} if self.objetivo == "quantile" else {}
        # Configuración conservadora: la demanda mensual de este dataset es muy
        # ruidosa, y hojas chicas con muchos ejemplos (min_child_samples) evitan
        # que los árboles memoricen pedidos sueltos. Medida en la selección 2015.
        return LGBMRegressor(
            objective=self.objetivo,
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=200,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            random_state=cfg.SEMILLA,
            verbose=-1,
            **extra,
        )


def _preprocesador(X: pd.DataFrame):
    """Imputación con indicador y estandarización; one-hot para las categóricas."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    categoricas = [c for c in X.columns if c in F.CATEGORICAS or c in ("mes_objetivo", "semana_objetivo", "paso")]
    numericas = [c for c in X.columns if c not in categoricas]
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputar", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                        ("estandarizar", StandardScaler()),
                    ]
                ),
                numericas,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
        ]
    )


class RidgeGlobal(ModeloGlobal):
    """Regresión lineal con regularización L2, sobre las mismas variables."""

    nombre = "Ridge"

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def _estimador(self, X: pd.DataFrame):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline

        return Pipeline([("pre", _preprocesador(X)), ("ridge", Ridge(alpha=self.alpha))])

    def _kwargs_peso(self, meta: pd.DataFrame) -> dict:
        return {"ridge__sample_weight": self._pesos(meta)}


def clasificador_logistico(X: pd.DataFrame, balanceado: bool = False):
    """Regresión logística para «¿habrá algún pedido en t + 2?» (anexo de 02)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("pre", _preprocesador(X)),
            (
                "logit",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced" if balanceado else None, random_state=cfg.SEMILLA
                ),
            ),
        ]
    )


def modelos_regresion(k_media: int, objetivo_lgbm: str) -> list[Modelo]:
    """Los siete modelos del backtest, en el orden en que se reportan."""
    return [
        Naive(),
        NaiveEstacional(),
        MediaMovil(k_media),
        CrostonSBA(),
        ETS(),
        RidgeGlobal(),
        LightGBM(objetivo_lgbm),
    ]
