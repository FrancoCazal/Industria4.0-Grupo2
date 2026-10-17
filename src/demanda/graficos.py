"""Estilo común de las figuras y colores fijos por modelo."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from demanda import config as cfg

#: Grises para las referencias simples, color para lo que se defiende.
COLORES = {
    "Naive": "#b0b0b0",
    "Naive estacional": "#8c8c8c",
    "Media móvil": "#5f6b7a",
    "Croston SBA": "#3d4a5c",
    "ETS": "#d08c2c",
    "Ridge": "#4f9a94",
    "LightGBM": "#c0392b",
}
DEPOSITOS = {"Whse_J": "#1f4e79", "Whse_S": "#4f9a94", "Whse_C": "#d08c2c", "Whse_A": "#8c8c8c"}


def color(modelo: str) -> str:
    """Color del modelo; las medias móviles comparten color sin importar la ventana."""
    for clave, c in COLORES.items():
        if modelo.startswith(clave):
            return c
    return "#333333"


def estilo() -> None:
    """Aplica el estilo de todas las figuras del repo."""
    plt.rcParams.update(
        {
            "figure.figsize": (10, 4.5),
            "figure.dpi": 110,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.frameon": False,
            "font.size": 10,
        }
    )


def miles(eje, en: str = "y") -> None:
    """Formatea un eje con separador de miles con punto, como se lee en Paraguay."""
    fmt = FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", "."))
    (eje.yaxis if en == "y" else eje.xaxis).set_major_formatter(fmt)


def millones(eje, en: str = "y") -> None:
    """Formatea un eje en millones de unidades."""
    fmt = FuncFormatter(lambda v, _: f"{v / 1e6:,.0f}".replace(",", "."))
    (eje.yaxis if en == "y" else eje.xaxis).set_major_formatter(fmt)


def porcentaje(eje, en: str = "y", decimales: int = 0) -> None:
    """Formatea un eje de proporciones como porcentaje con coma decimal."""
    fmt = FuncFormatter(lambda v, _: f"{v * 100:.{decimales}f} %".replace(".", ","))
    (eje.yaxis if en == "y" else eje.xaxis).set_major_formatter(fmt)


def guardar(fig, nombre: str) -> None:
    """Guarda la figura en `resultados/figuras/` con un nombre fijo."""
    cfg.FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.FIGURAS / f"{nombre}.png", dpi=150, bbox_inches="tight")


def puntos(x: float, decimales: int = 1) -> str:
    """Diferencia entre dos proporciones: 0,038 -> '3,8 puntos'."""
    return f"{x * 100:.{decimales}f} puntos".replace(".", ",")


def pct(x: float, decimales: int = 1) -> str:
    """0,1234 -> '12,3 %'. Para imprimir conclusiones con las cifras del notebook."""
    return f"{x * 100:.{decimales}f} %".replace(".", ",")


def num(x: float, decimales: int = 0) -> str:
    """1234567,8 -> '1.234.568'."""
    s = f"{x:,.{decimales}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")
