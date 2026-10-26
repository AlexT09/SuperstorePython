"""tabs/prediccion.py — Predicción interactiva: carga model/artifacts/model.pkl y predice en tiempo real.

El usuario elige categoría, región y mes. El formulario se llena con la historia real de esa serie
(promedio de 12 meses, mismo mes del año anterior, mes anterior y promedio de 3 meses) y cada valor
se puede editar para simular escenarios. Con cada cambio, el modelo entrenado predice las ventas del mes.
"""
from functools import lru_cache

import dash_bootstrap_components as dbc
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from common import (ART, BRAND_DARK, INK, PROY, aviso_sin_modelo, insight, kpi, load_artifact,
                    load_mensual, page_header, soft_card, style_fig, usd)
from model.features import FEATURES, agregar_variables

CFG = {"displayModeBar": False}
MODEL_PATH = ART / "model.pkl"
CATS = ["Furniture", "Office Supplies", "Technology"]
REGIONS = ["Central", "East", "South", "West"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre"]
FUTURO = pd.Timestamp("2018-01-01")  # primer mes sin ventas reales

# Campos del formulario: (id del input, columna de la historia, etiqueta)
CAMPOS = [
    ("pred-prom12", "prom_12m", "Promedio mensual de los últimos 12 meses"),
    ("pred-lag12", "lag_12", "Ventas del mismo mes del año anterior"),
    ("pred-lag1", "lag_1", "Ventas del mes anterior"),
    ("pred-prom3", "prom_3m", "Promedio de los últimos 3 meses"),
]


@lru_cache(maxsize=1)
def cargar_modelo():
    """Carga el pipeline entrenado (preprocesamiento + modelo) una sola vez."""
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def historia() -> pd.DataFrame:
    """Variables del modelo para cada serie y mes, incluido enero 2018 (sin ventas reales)."""
    m = load_mensual()[["Category", "Region", "Mes", "Sales"]]
    futuro = m[["Category", "Region"]].drop_duplicates().assign(Mes=FUTURO, Sales=np.nan)
    h = agregar_variables(pd.concat([m, futuro], ignore_index=True))
    return h.dropna(subset=["prom_12m", "lag_12", "lag_1", "prom_3m"])


def _etiqueta_mes(mes: pd.Timestamp) -> str:
    return f"{MESES[mes.month - 1].capitalize()} {mes.year}"


def layout():
    if not MODEL_PATH.exists():
        return html.Div([page_header("Predicción"), aviso_sin_modelo()])
    resumen = load_artifact("resumen.json")
    meses = sorted(historia().Mes.unique(), reverse=True)
    opciones_mes = [{"label": _etiqueta_mes(pd.Timestamp(m)) + (" (futuro)" if m == FUTURO else ""),
                     "value": pd.Timestamp(m).strftime("%Y-%m-%d")} for m in meses]

    def campo(id_, etiqueta):
        return dbc.Col([
            html.Label(etiqueta, className="small fw-bold", htmlFor=id_),
            dbc.InputGroup([dbc.InputGroupText("$"),
                            dbc.Input(id=id_, type="number", min=0, step=1, debounce=True)]),
        ], md=6, className="mb-3")

    return html.Div([
        page_header("Predicción", f"Ventas de un mes con el modelo entrenado ({resumen['mejor_modelo']}, "
                    "cargado desde model.pkl)"),
        dbc.Row([
            dbc.Col(soft_card("Datos de entrada", [
                dbc.Row([
                    dbc.Col([html.Label("Categoría", className="small fw-bold"),
                             dcc.Dropdown(CATS, "Technology", id="pred-cat", clearable=False)], md=4, className="mb-3"),
                    dbc.Col([html.Label("Región", className="small fw-bold"),
                             dcc.Dropdown(REGIONS, "West", id="pred-reg", clearable=False)], md=4, className="mb-3"),
                    dbc.Col([html.Label("Mes a predecir", className="small fw-bold"),
                             dcc.Dropdown(opciones_mes, opciones_mes[0]["value"], id="pred-mes", clearable=False)],
                            md=4, className="mb-3"),
                ]),
                html.Hr(className="my-2"),
                html.P("Historia de la serie antes de ese mes (USD). Se llena con los datos reales; "
                       "cámbiela para simular otro escenario.", className="small text-muted"),
                dbc.Row([campo(id_, etiqueta) for id_, _, etiqueta in CAMPOS]),
                insight("Al elegir otra categoría, región o mes, el formulario vuelve a los valores reales."),
            ], "blue"), lg=6, className="mb-3"),
            dbc.Col(soft_card("Resultado", [
                dbc.Row(id="pred-kpis"),
                dcc.Graph(id="pred-fig", config=CFG),
                html.Div(id="pred-nota"),
            ], "lavender"), lg=6, className="mb-3"),
        ]),
        soft_card("Cómo predice el modelo", html.Ul([
            html.Li("El modelo no predice dólares directamente, sino un índice: cuántas veces el promedio de los "
                    "últimos 12 meses venderá ese mes. Las ventas = índice × promedio de 12 meses."),
            html.Li("Las demás ventas del formulario se dividen por ese promedio, y el mes se codifica como seno y "
                    "coseno para que diciembre quede junto a enero."),
            html.Li("Categoría y región entran con one-hot encoding; todo pasa por el mismo Pipeline con el que "
                    "se entrenó (model/features.py)."),
        ], className="mb-0"), "peach", height_full=False),
    ])


@callback([Output(id_, "value") for id_, _, _ in CAMPOS],
          Input("pred-cat", "value"), Input("pred-reg", "value"), Input("pred-mes", "value"))
def llenar_formulario(cat, reg, mes):
    """Llena el formulario con la historia real de la serie elegida."""
    fila = _fila(cat, reg, mes)
    return [round(float(fila[col]), 2) for _, col, _ in CAMPOS]


def _fila(cat, reg, mes):
    h = historia()
    return h[(h.Category == cat) & (h.Region == reg) & (h.Mes == pd.Timestamp(mes))].iloc[0]


@callback(Output("pred-kpis", "children"), Output("pred-fig", "figure"), Output("pred-nota", "children"),
          Input("pred-cat", "value"), Input("pred-reg", "value"), Input("pred-mes", "value"),
          [Input(id_, "value") for id_, _, _ in CAMPOS])
def predecir(cat, reg, mes, prom12, lag12, lag1, prom3):
    """Arma las variables con los valores del formulario y predice con model.pkl."""
    valores = [prom12, lag12, lag1, prom3]
    if any(v is None or v < 0 for v in valores) or not prom12:
        aviso = dbc.Alert("Complete los cuatro valores (el promedio de 12 meses debe ser mayor que cero).",
                          color="warning", className="mb-0")
        return [], style_fig(go.Figure(), 300), aviso

    mes_ts = pd.Timestamp(mes)
    X = pd.DataFrame([{
        "Category": cat, "Region": reg,
        "r_lag_12": lag12 / prom12, "r_lag_1": lag1 / prom12, "r_prom_3m": prom3 / prom12,
        "mes_sin": np.sin(2 * np.pi * mes_ts.month / 12), "mes_cos": np.cos(2 * np.pi * mes_ts.month / 12),
    }])[FEATURES]
    indice = float(cargar_modelo().predict(X)[0])
    pred = max(indice * prom12, 0.0)

    real = _fila(cat, reg, mes).Sales
    tiene_real = not pd.isna(real)
    kpis = [
        dbc.Col(kpi("Ventas predichas", usd(pred), _etiqueta_mes(mes_ts)), md=4, className="mb-3"),
        dbc.Col(kpi("Índice", f"{indice:.2f}×", "del promedio de 12 meses"), md=4, className="mb-3"),
        dbc.Col(kpi("Real", usd(real), f"error: {(pred / real - 1) * 100:+.0f} %") if tiene_real and real > 0
                else kpi("Real", "—", "mes futuro, aún sin ventas"), md=4, className="mb-3"),
    ]

    nombres = ["Promedio 12 meses", "Mismo mes año ant.", "Mes anterior", "Promedio 3 meses", "Predicción"]
    colores = [BRAND_DARK] * 4 + [PROY]
    fig = go.Figure(go.Bar(x=nombres, y=valores + [pred], marker_color=colores, marker_line_color="rgba(0,0,0,0)",
                           marker_line_width=1, text=[usd(v) for v in valores + [pred]], textposition="outside",
                           hovertemplate="%{x}: %{y:$,.0f}<extra></extra>"))
    if tiene_real:
        fig.add_hline(y=real, line_dash="dot", line_color=INK,
                      annotation_text=f"Real: {usd(real)}", annotation_position="top left")
    fig.update_yaxes(title="USD", tickprefix="$", tickformat=",.0f", rangemode="tozero")
    fig = style_fig(fig, 320)
    fig.update_layout(showlegend=False, margin=dict(l=50, r=20, t=30, b=40))

    if mes_ts >= FUTURO:
        nota = insight("Enero 2018 es un mes futuro: el modelo nunca vio sus ventas, es una predicción real.")
    else:
        nota = insight("El modelo final se entrenó con 2015–2017, así que ya conoce este mes. Para medir su "
                       "error sin ventaja, vea la prueba de 2017 en la pestaña Comparación.")
    return kpis, fig, nota
