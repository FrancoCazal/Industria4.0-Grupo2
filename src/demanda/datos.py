"""Descarga, limpieza y armado del panel de demanda.

La limpieza está escrita también, paso a paso, en el notebook 01, porque es lo
que se muestra en vivo. Esta versión existe para que el notebook 03 la pueda
repetir sobre datos degradados; 01 verifica que las dos den lo mismo.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from demanda import config as cfg

# Umbrales de Syntetos y Boylan (2005) para clasificar la intermitencia.
ADI_CORTE = 1.32
CV2_CORTE = 0.49


# --------------------------------------------------------------------------
# Descarga y lectura
# --------------------------------------------------------------------------
def descargar_crudo(destino: Path = cfg.DATA_RAW) -> Path:
    """Devuelve la ruta del CSV crudo, descargándolo de Kaggle si hace falta.

    `kagglehub` baja datasets públicos sin credenciales. El archivo se copia a
    `data/raw/` para que el repo no dependa de la caché del usuario.
    """
    destino.mkdir(parents=True, exist_ok=True)
    archivo = destino / cfg.ARCHIVO_CRUDO
    if archivo.exists():
        return archivo
    import kagglehub

    carpeta = Path(kagglehub.dataset_download(cfg.KAGGLE_DATASET))
    shutil.copy2(carpeta / cfg.ARCHIVO_CRUDO, archivo)
    return archivo


def leer_crudo(archivo: Path) -> pd.DataFrame:
    """Lee el CSV con todas las columnas como texto, sin interpretar nada."""
    return pd.read_csv(archivo, dtype=str)


# --------------------------------------------------------------------------
# Limpieza
# --------------------------------------------------------------------------
def a_numero(texto: pd.Series) -> pd.Series:
    """Convierte `Order_Demand` a número: `(100)` es -100."""
    t = texto.str.strip()
    negativo = t.str.startswith("(")
    valor = pd.to_numeric(t.str.replace(r"[()]", "", regex=True))
    return valor.where(~negativo, -valor)


def limpiar(crudo: pd.DataFrame) -> pd.DataFrame:
    """Limpieza completa del archivo crudo.

    Devuelve una fila por pedido con `producto`, `deposito`, `categoria`,
    `fecha`, `mes` y `cantidad` (negativa si es devolución). Descarta las filas
    sin fecha y las de fuera de 2012-2016.
    """
    df = pd.DataFrame(
        {
            "producto": crudo["Product_Code"].str.strip(),
            "deposito": crudo["Warehouse"].str.strip(),
            "categoria": crudo["Product_Category"].str.strip(),
            "fecha": pd.to_datetime(crudo["Date"], format="%Y/%m/%d"),
            "cantidad": a_numero(crudo["Order_Demand"]),
        }
    )
    df = df.dropna(subset=["fecha"])
    df["mes"] = df["fecha"].dt.to_period("M")
    df = df[(df["mes"] >= cfg.MES_INICIO) & (df["mes"] <= cfg.MES_FIN)]
    return df.reset_index(drop=True)


def guardar_limpio(limpio: pd.DataFrame, archivo: Path = cfg.ARCHIVO_LIMPIO) -> None:
    """Guarda el dataset limpio. El mes se recalcula al leer, no se guarda."""
    archivo.parent.mkdir(parents=True, exist_ok=True)
    limpio.drop(columns="mes").to_parquet(archivo, index=False)


def leer_limpio(archivo: Path = cfg.ARCHIVO_LIMPIO) -> pd.DataFrame:
    """Lee el dataset limpio que escribe 01."""
    if not archivo.exists():
        raise FileNotFoundError(f"Falta {archivo.name}: corré primero el notebook 01.")
    limpio = pd.read_parquet(archivo)
    limpio["mes"] = limpio["fecha"].dt.to_period("M")
    return limpio


def id_serie(producto: pd.Series, deposito: pd.Series) -> pd.Series:
    """Identificador de la serie producto × depósito."""
    return producto + "|" + deposito


# --------------------------------------------------------------------------
# Panel
# --------------------------------------------------------------------------
@dataclass
class Panel:
    """Demanda en forma de matriz series × períodos.

    `Y[i, j]` es la demanda de la serie `i` en el período `j`. Vale NaN antes del
    primer pedido de la serie (la serie todavía no existe) y cero en los períodos
    sin pedidos posteriores. Con `p = 1` cada período es un mes; con `p = 4`, una
    semana del mes (días 1-7, 8-14, 15-21 y 22 a fin de mes).
    """

    Y: np.ndarray
    series: pd.DataFrame
    p: int
    mes0: pd.Period = cfg.MES_INICIO

    @property
    def n_meses(self) -> int:
        return self.Y.shape[1] // self.p

    def indice_mes(self, mes: pd.Period) -> int:
        return (mes - self.mes0).n

    def fin_de_mes(self, mes: pd.Period) -> int:
        """Índice del último período del mes, que es el origen de un pronóstico."""
        return self.indice_mes(mes) * self.p + self.p - 1

    def mes_de_periodo(self, j: np.ndarray | int) -> np.ndarray:
        """Mes del año (1-12) de cada período."""
        return (np.asarray(j) // self.p + self.mes0.month - 1) % 12 + 1

    def mensual(self) -> np.ndarray:
        """Suma los `p` períodos de cada mes. NaN solo si el mes entero es NaN."""
        if self.p == 1:
            return self.Y
        n, t = self.Y.shape
        bloques = self.Y.reshape(n, t // self.p, self.p)
        todo_nan = np.isnan(bloques).all(axis=2)
        suma = np.nansum(bloques, axis=2)
        suma[todo_nan] = np.nan
        return suma

    def con_Y(self, Y: np.ndarray) -> Panel:
        return Panel(Y=Y, series=self.series, p=self.p, mes0=self.mes0)


def _periodo(df: pd.DataFrame, p: int) -> np.ndarray:
    """Índice del período de cada fila: meses (p = 1) o semanas del mes (p = 4)."""
    f = df["fecha"]
    base = cfg.MES_INICIO.year * 12 + cfg.MES_INICIO.month
    meses = (f.dt.year * 12 + f.dt.month - base).to_numpy()
    if p == 1:
        return meses
    semana = np.minimum((df["fecha"].dt.day.to_numpy() - 1) // 7, 3)
    return meses * 4 + semana


def catalogo_series(limpio: pd.DataFrame) -> pd.DataFrame:
    """Series con al menos un pedido positivo en 2012-2016, en orden fijo."""
    pos = limpio[limpio["cantidad"] > 0]
    cat = (
        pos.groupby(["producto", "deposito"], as_index=False)["categoria"]
        .first()
        .sort_values(["producto", "deposito"])
        .reset_index(drop=True)
    )
    cat.insert(0, "serie", id_serie(cat["producto"], cat["deposito"]))
    return cat


def armar_panel(
    limpio: pd.DataFrame, *, p: int = 1, series: pd.DataFrame | None = None
) -> Panel:
    """Arma el panel de demanda (suma de pedidos positivos) por período.

    `series` fija el catálogo, para que todos los escenarios del experimento
    tengan las mismas filas aunque a alguno le falten registros. Una serie sin
    ningún pedido positivo queda entera en NaN.
    """
    if series is None:
        series = catalogo_series(limpio)
    n_periodos = (cfg.MES_FIN - cfg.MES_INICIO).n * p + p
    pos = limpio[limpio["cantidad"] > 0]
    fila = pd.Series(np.arange(len(series)), index=series["serie"])
    i = fila.reindex(id_serie(pos["producto"], pos["deposito"])).to_numpy()
    ok = ~np.isnan(i)
    i = i[ok].astype(int)
    j = _periodo(pos, p)[ok]

    Y = np.zeros((len(series), n_periodos))
    np.add.at(Y, (i, j), pos["cantidad"].to_numpy()[ok])

    # Antes del primer pedido la serie no existe: NaN, no cero.
    tiene = Y > 0
    primero = np.where(tiene.any(axis=1), tiene.argmax(axis=1), n_periodos)
    Y[np.arange(n_periodos)[None, :] < primero[:, None]] = np.nan
    return Panel(Y=Y, series=series.reset_index(drop=True), p=p)


def panel_largo(limpio: pd.DataFrame, panel: Panel) -> pd.DataFrame:
    """Panel mensual en formato largo, con devoluciones y cantidad de pedidos.

    Una fila por serie y mes, desde el primer pedido de la serie hasta 2016-12.
    """
    assert panel.p == 1, "el formato largo es mensual"
    meses = pd.period_range(cfg.MES_INICIO, cfg.MES_FIN, freq="M")
    largo = pd.DataFrame(panel.Y, columns=meses)
    largo["serie"] = panel.series["serie"].to_numpy()
    largo = largo.melt(id_vars="serie", var_name="mes", value_name="demanda").dropna()
    largo["mes"] = largo["mes"].astype("period[M]")

    s = id_serie(limpio["producto"], limpio["deposito"])
    extra = (
        limpio.assign(
            serie=s,
            devoluciones=(-limpio["cantidad"]).clip(lower=0),
            pedidos=(limpio["cantidad"] > 0).astype(int),
        )
        .groupby(["serie", "mes"], as_index=False)[["devoluciones", "pedidos"]]
        .sum()
    )
    largo = largo.merge(extra, on=["serie", "mes"], how="left").fillna(
        {"devoluciones": 0, "pedidos": 0}
    )
    largo = largo.merge(panel.series, on="serie", how="left")
    largo["pedidos"] = largo["pedidos"].astype(int)
    cols = ["serie", "producto", "deposito", "categoria", "mes", "demanda", "devoluciones", "pedidos"]
    return largo[cols].sort_values(["serie", "mes"]).reset_index(drop=True)


def guardar_panel(largo: pd.DataFrame, archivo: Path = cfg.ARCHIVO_PANEL) -> None:
    """Guarda el panel largo con el mes como fecha (primer día del mes)."""
    archivo.parent.mkdir(parents=True, exist_ok=True)
    largo.assign(mes=largo["mes"].dt.to_timestamp()).to_parquet(archivo, index=False)


def leer_panel(archivo: Path = cfg.ARCHIVO_PANEL) -> pd.DataFrame:
    """Lee el panel largo que escribe 01, con el mes como período mensual."""
    if not archivo.exists():
        raise FileNotFoundError(f"Falta {archivo.name}: corré primero el notebook 01.")
    largo = pd.read_parquet(archivo)
    largo["mes"] = largo["mes"].dt.to_period("M")
    return largo


def panel_desde_largo(largo: pd.DataFrame) -> Panel:
    """Reconstruye el `Panel` mensual a partir del parquet que escribe 01."""
    series = (
        largo[["serie", "producto", "deposito", "categoria"]]
        .drop_duplicates("serie")
        .sort_values(["producto", "deposito"])
        .reset_index(drop=True)
    )
    meses = pd.period_range(cfg.MES_INICIO, cfg.MES_FIN, freq="M")
    mes = pd.PeriodIndex(largo["mes"], freq="M")
    ancho = (
        largo.assign(mes=mes)
        .pivot(index="serie", columns="mes", values="demanda")
        .reindex(index=series["serie"], columns=meses)
    )
    return Panel(Y=ancho.to_numpy(dtype="float64"), series=series, p=1)


# --------------------------------------------------------------------------
# Segmentación (solo para describir y segmentar resultados, nunca como feature)
# --------------------------------------------------------------------------
def clase_abc(volumen: pd.Series, cortes: tuple[float, float] = (0.80, 0.95)) -> pd.Series:
    """A: el primer 80 % del volumen acumulado; B: hasta 95 %; C: el resto."""
    orden = volumen.sort_values(ascending=False)
    acumulado = orden.cumsum() / orden.sum()
    # La serie que cruza el corte pertenece a la clase del corte.
    previo = acumulado.shift(fill_value=0)
    clase = np.where(previo < cortes[0], "A", np.where(previo < cortes[1], "B", "C"))
    return pd.Series(clase, index=orden.index).reindex(volumen.index)


def intermitencia(Y: np.ndarray) -> pd.DataFrame:
    """ADI y CV² de cada serie, y su clase según Syntetos y Boylan (2005).

    ADI: intervalo medio entre meses con pedido. CV²: variabilidad del tamaño de
    los pedidos no nulos. Se calcula sobre la vida observada de cada serie.
    """
    filas = []
    for y in Y:
        y = y[~np.isnan(y)]
        nz = y[y > 0]
        adi = len(y) / len(nz) if len(nz) else np.nan
        cv2 = (nz.std() / nz.mean()) ** 2 if len(nz) > 1 else 0.0
        filas.append((adi, cv2))
    out = pd.DataFrame(filas, columns=["adi", "cv2"])
    regular = out["adi"] < ADI_CORTE
    estable = out["cv2"] < CV2_CORTE
    out["intermitencia"] = np.select(
        [regular & estable, regular & ~estable, ~regular & estable],
        ["suave", "errática", "intermitente"],
        default="irregular",
    )
    return out


def degradar(limpio: pd.DataFrame, fraccion: float, semilla: int = cfg.SEMILLA) -> pd.DataFrame:
    """Borra al azar una fracción de los registros: ventas que nunca se cargaron."""
    if fraccion <= 0:
        return limpio
    rng = np.random.default_rng(semilla)
    queda = rng.random(len(limpio)) >= fraccion
    return limpio[queda].reset_index(drop=True)
