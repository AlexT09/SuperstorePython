"""
common.py
Utilidades compartidas: rutas, carga de datos, paleta de colores y helpers de UI.
Las pestañas (tabs/*.py) solo dependen de este módulo, nunca entre sí.
"""
import json
from functools import lru_cache
from pathlib import Path

import dash_bootstrap_components as dbc
import pandas as pd
from dash import html

# ---------------------------------------------------------------- rutas
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "superstore_transformado.csv"
MENSUAL_PATH = ROOT / "data" / "ventas_mensuales.csv"
ART = ROOT / "model" / "artifacts"  # salidas de model/train_model.py

# ---------------------------------------------------------------- colores
# Tema oscuro: fondo azul petróleo, tarjetas de vidrio y texto blanco.
# Cada color tiene una función fija en toda la app.
INK = "#ffffff"                        # texto principal
INK_SOFT = "rgba(255,255,255,.78)"     # texto secundario
INK_MUTED = "rgba(255,255,255,.58)"    # notas y ejes
GRID = "rgba(255,255,255,.10)"         # cuadrícula de los gráficos
GLASS = "rgba(255,255,255,.12)"        # rellenos suaves (bandas, encabezados de tabla)
SURFACE = "#123240"                    # fondo de referencia de los gráficos

REAL = "#4a93e6"         # serie: ventas reales / color principal
PROY = "#f0b851"         # serie: proyección
PROY_DARK = "#c98500"
NEG = "#f08a7e"          # caídas, residuos negativos
POS = "#8fd3b0"          # crecimientos, mejor resultado
REF = "rgba(255,255,255,.55)"  # referencia y líneas auxiliares

# Escalas continuas (mapas de calor)
SEQ_SCALE = [[0, "#3d7f9c"], [0.35, "#5fa3c0"], [0.7, "#8cc6de"], [1, "#c9e8f4"]]
DIV_SCALE = [[0, "#e66767"], [0.5, "#5e7580"], [1, "#4a93e6"]]

# Nombres usados por las pestañas
BRAND = REAL
BRAND_DARK = "#3d7f9c"
BRAND_DARKER = "#c9e8f4"   # el tono más claro: máximo contraste sobre el fondo oscuro
BRAND_LIGHT = GLASS
TEXT = INK
MARK = PROY  # puntos de referencia (p. ej. la media en un boxplot)

PASTEL = {
    "blue": "#c9e8f4",
    "green": "#8cc6de",
    "peach": "#5fa3c0",
    "pink": "#4a93e6",
    "yellow": "#f0b851",
    "lavender": "#b89cf0",
}
COLORWAY = ["#4a93e6", "#5fc4b8", "#b89cf0", "#f0b851", "#8cc6de", "#f08a7e"]
ACCENT = {
    "blue": REAL,
    "pink": PROY,
    "green": POS,
    "peach": "#5fc4b8",
    "lavender": "#b89cf0",
    "yellow": PROY,
}
# Colores de categoría
CATEGORY_COLORS = {
    "Furniture": "#5fc4b8",
    "Office Supplies": "#4a93e6",
    "Technology": "#b89cf0",
}
CATEGORY_ACCENT = CATEGORY_COLORS
@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    """Carga el dataset transformado (salida de data/generate_data.py)."""
    return pd.read_csv(DATA_PATH, parse_dates=["Order_Date", "Ship Date"])


@lru_cache(maxsize=1)
def load_mensual() -> pd.DataFrame:
    """Ventas por categoría × región × mes (salida de data/generate_data.py)."""
    return pd.read_csv(MENSUAL_PATH, parse_dates=["Mes"])


# ---------------------------------------------------------------- resultados del modelo
# La app nunca entrena: solo lee lo que deja model/train_model.py en model/artifacts/.
REFERENCIA = "Referencia (año anterior)"


def modelo_listo() -> bool:
    return (ART / "resumen.json").exists()


@lru_cache(maxsize=None)
def load_artifact(name: str):
    """Lee un archivo de model/artifacts/ (CSV como DataFrame, JSON como dict)."""
    path = ART / name
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    df = pd.read_csv(path)
    if "Mes" in df:
        df["Mes"] = pd.to_datetime(df["Mes"])
    return df


def mejores_por_modelo() -> pd.DataFrame:
    """Resultados de los 7 modelos (sin la referencia), ordenados por MAE en la prueba."""
    r = load_artifact("resultados_modelos.csv")
    return r[r.modelo != REFERENCIA].sort_values("MAE")


def aviso_sin_modelo():
    """Mensaje para las pestañas del modelo cuando aún no se ha entrenado."""
    return soft_card("Modelo no entrenado", html.P([
        "Ejecute ", html.Code("python model/train_model.py"), " para generar los resultados y vuelva a "
        "cargar la página."], className="mb-0"), height_full=False)


def usd(x, dec=0):
    return f"${x:,.{dec}f}"


def style_fig(fig, height=360, title=None):
    """Aplica el tema oscuro (fondo transparente, texto blanco) a cualquier figura de Plotly."""
    fig.update_layout(
        template="plotly_dark",
        height=height,
        title=dict(text=title, font=dict(size=15, color=INK)) if title else None,
        font=dict(family="Inter, Segoe UI, sans-serif", color=INK_SOFT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=COLORWAY,
        margin=dict(l=40, r=20, t=50 if title else 20, b=40),
        legend=dict(orientation="h", y=-0.2, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#0b2531", bordercolor="rgba(255,255,255,.25)", font=dict(color=INK)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


# ---------------------------------------------------------------- helpers de UI
def page_header(title: str, subtitle: str = "", icon: str = "") -> html.Div:
    return html.Div(
        [html.H2(f"{icon} {title}".strip(), className="page-title"),
         html.P(subtitle, className="page-subtitle")],
        className="mb-4",
    )


def soft_card(title, body, color="blue", height_full=True):
    """Tarjeta blanca con un acento de color sutil (borde) según `color`."""
    classes = f"soft-card accent-{color}" + (" h-100" if height_full else "")
    return dbc.Card(
        [dbc.CardHeader(title, className="soft-card-header"),
         dbc.CardBody(body)],
        className=classes,
    )


def kpi(label, value, detail=""):
    """Cifra destacada: etiqueta, valor grande y una línea de contexto."""
    return html.Div([html.Div(label, className="kpi-label"), html.Div(value, className="kpi-value"),
                     html.Div(detail, className="kpi-detail")], className="kpi h-100")


def insight(children):
    """Párrafo pequeño y discreto para interpretar un gráfico o dato (estilo uniforme en todo el dashboard)."""
    return html.P(children, className="insight-text")

