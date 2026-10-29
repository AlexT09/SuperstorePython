"""
common.py
Utilidades compartidas por los notebooks del libro: rutas, carga de datos y resultados del modelo,
paleta y estilo de figuras. Es el common.py de la app Dash (carpeta superior), sin los helpers de
interfaz y con tema claro.
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
from IPython.display import HTML, Markdown, display

# ---------------------------------------------------------------- rutas
# El libro vive en libro/ y usa los mismos datos y el mismo modelo que la app Dash (raíz del repo)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # para "from model.features import ..." en los notebooks
DATA_PATH = ROOT / "data" / "superstore_transformado.csv"
MENSUAL_PATH = ROOT / "data" / "ventas_mensuales.csv"
ART = ROOT / "model" / "artifacts"  # salidas de model/train_model.py

# ---------------------------------------------------------------- colores
# Cada color tiene una función fija en todo el libro (los mismos papeles que en la app Dash).
TEXT = "#2f3247"
GRID = "#EEF0F6"
BRAND = "#2E6F95"
BRAND_LIGHT = "#EAF2F7"

REAL = "#2E6F95"         # serie: ventas reales / color principal
PROY = "#E0962B"         # serie: proyección
NEG = "#D1574A"          # caídas
POS = "#2CA58D"          # crecimientos, mejor resultado
REF = "#8A8FA3"          # referencia y líneas auxiliares
MARK = "#153B4E"         # puntos de referencia (p. ej. la media en un boxplot)

SEQ_SCALE = [[0, "#F4F8FB"], [0.5, "#8FBBD1"], [1, "#1F5570"]]
COLORWAY = ["#2E6F95", "#2CA58D", "#8E6CD8", "#E0962B", "#63A0BC", "#D1574A"]
PASTEL = {"blue": "#63A0BC", "green": "#2CA58D", "peach": "#8FBBD1", "pink": "#2E6F95"}

# Un tono distinto por categoría (el mismo en todas las gráficas)
CATEGORY_COLORS = {
    "Furniture": "#2CA58D",
    "Office Supplies": "#4A93E6",
    "Technology": "#8E6CD8",
}

REFERENCIA = "Referencia (año anterior)"


@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    """Carga el dataset transformado (salida de data/generate_data.py)."""
    return pd.read_csv(DATA_PATH, parse_dates=["Order_Date", "Ship Date"])


@lru_cache(maxsize=1)
def load_mensual() -> pd.DataFrame:
    """Ventas por categoría × región × mes (salida de data/generate_data.py)."""
    return pd.read_csv(MENSUAL_PATH, parse_dates=["Mes"])


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


def usd(x, dec=0):
    return f"${x:,.{dec}f}"


def md(texto):
    """Muestra texto Markdown calculado (cifras que salen de los resultados del modelo)."""
    display(Markdown(texto.replace("$", r"\$")))


def kpis(items):
    """Fila de cifras destacadas: lista de (etiqueta, valor, detalle)."""
    celdas = "".join(
        f'<div style="flex:1 1 180px;border:1px solid #DCE6EE;border-radius:10px;padding:10px 14px;'
        f'background:{BRAND_LIGHT}">'
        f'<div style="font-size:.8rem;color:#5b6075">{label}</div>'
        f'<div style="font-size:1.45rem;font-weight:700;color:{BRAND}">{valor}</div>'
        f'<div style="font-size:.8rem;color:#5b6075">{detalle}</div></div>'
        for label, valor, detalle in items)
    display(HTML(f'<div style="display:flex;flex-wrap:wrap;gap:10px;margin:8px 0 16px">{celdas}</div>'))


def tabla(df):
    """Tabla sin índice, con un estilo uniforme."""
    return (df.style.hide(axis="index")
              .set_table_styles([{"selector": "th, td", "props": "padding: 5px 12px; text-align: left"}]))


def style_fig(fig, height=360, title=None):
    """Aplica el estilo común a cualquier figura de Plotly."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=dict(text=title, font=dict(size=15, color=TEXT)) if title else None,
        font=dict(family="Nunito, Segoe UI, sans-serif", color=TEXT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=COLORWAY,
        margin=dict(l=40, r=20, t=50 if title else 20, b=40),
        legend=dict(orientation="h", y=-0.2),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def show(fig, height=None):
    """Muestra una figura de Plotly en el libro.

    Se inserta como HTML sin plotly.js: el libro lo carga una sola vez (_config.yml → html_js_files).
    Así la figura es interactiva en el HTML generado sin repetir ~4 MB de librería por gráfica.
    """
    if height is not None:
        style_fig(fig, height)
    display(HTML(fig.to_html(include_plotlyjs=False, full_html=False, config={"displayModeBar": False})))
