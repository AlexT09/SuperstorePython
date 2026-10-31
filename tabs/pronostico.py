"""tabs/pronostico.py — Pronóstico animado de 2018 en una sola pantalla.

Barras trimestrales tipo holograma, una línea punteada con las ventas de los últimos 12 meses y un círculo por
año (el total del año). Al presionar "Predecir 2018" se dibuja la historia, suben las barras de 2018 y el último
punto se enciende con el total proyectado. La animación corre en el navegador (assets/pronostico.js); aquí solo
se preparan los datos.
"""
import numpy as np
import pandas as pd
from dash import Input, Output, callback, clientside_callback, dcc, html

from common import aviso_sin_modelo, load_artifact, load_mensual, modelo_listo

TODAS = "Todas"
CATS = ["Furniture", "Office Supplies", "Technology"]
REGIONS = ["Central", "East", "South", "West"]


def _filtrar(df, cat, reg):
    if cat != TODAS:
        df = df[df.Category == cat]
    if reg != TODAS:
        df = df[df.Region == reg]
    return df


def _error_mensual(cat, reg):
    """Error típico mensual (%) de la proyección a 12 meses, medido al proyectar 2017 con datos hasta 2016."""
    rec = _filtrar(load_artifact("prueba_recursiva_2017.csv"), cat, reg).groupby("Mes")[["Sales", "Proyeccion"]].sum()
    return float(np.mean(np.abs(rec.Sales - rec.Proyeccion)) / rec.Sales.mean() * 100)


def datos(cat, reg):
    """Serie trimestral (real 2014–2017 + pronóstico 2018) y los totales que muestra la animación."""
    real = _filtrar(load_mensual(), cat, reg).groupby("Mes").Sales.sum()
    proy = _filtrar(load_artifact("proyeccion_2018.csv"), cat, reg).groupby("Mes").Proyeccion.sum()
    serie = pd.concat([real, proy])
    trim = serie.groupby(serie.index.to_period("Q")).sum()
    movil = trim.rolling(4).sum()  # ventas de los últimos 12 meses; en cada T4 es el total del año
    err = _error_mensual(cat, reg)
    v17, v18 = float(real[real.index.year == 2017].sum()), float(proy.sum())
    return {
        "barras": [{"t": f"T{p.quarter} {p.year}", "anio": p.year, "q": p.quarter, "v": round(float(v), 2),
                    "pron": p.year == 2018, "m": None if np.isnan(movil[p]) else round(float(movil[p]), 2)}
                   for p, v in trim.items()],
        "v17": v17, "v18": v18, "crec": (v18 / v17 - 1) * 100, "err": err,
        "filtro": " · ".join(x for x in [cat if cat != TODAS else "Todas las categorías",
                                          reg if reg != TODAS else "todas las regiones"]),
    }


def layout():
    if not modelo_listo():
        return aviso_sin_modelo()
    sel = lambda id_, opts, etq: html.Label([html.Span(etq), dcc.Dropdown([TODAS] + opts, TODAS, id=id_,  # noqa: E731
                                                                        clearable=False)], className="pron-filtro")
    return html.Div([
        html.Div([
            html.Div([
                html.H2("Pronóstico de ventas 2018", className="pron-titulo"),
                html.P("Random Forest, proyección recursiva mes a mes. Elige un filtro y presiona Predecir.",
                       className="pron-sub"),
            ]),
            html.Div([
                sel("pron-cat", CATS, "Categoría"),
                sel("pron-reg", REGIONS, "Región"),
                html.Button([html.I(className="bi bi-stars"), html.Span("Predecir 2018")], id="pron-btn",
                            n_clicks=0, className="pron-btn"),
            ], className="pron-controles"),
        ], className="pron-cabecera"),

        html.Div([
            html.Div([html.Span("Ventas 2017", className="pron-k-etq"), html.Strong(id="pron-k17", className="pron-k-val"),
                      html.Span("real", className="pron-k-det")], className="pron-k"),
            html.Div([html.Span("Pronóstico 2018", className="pron-k-etq"),
                      html.Strong("—", id="pron-k18", className="pron-k-val pron-k-dorado"),
                      html.Span("presiona Predecir", id="pron-k18-det", className="pron-k-det")], className="pron-k destacado"),
            html.Div([html.Span("Crecimiento", className="pron-k-etq"), html.Strong("—", id="pron-kcrec", className="pron-k-val"),
                      html.Span("frente a 2017", className="pron-k-det")], className="pron-k"),
            html.Div([html.Span("Error típico mensual", className="pron-k-etq"), html.Strong(id="pron-kerr", className="pron-k-val"),
                      html.Span("medido al proyectar 2017", className="pron-k-det")], className="pron-k"),
        ], className="pron-kpis"),

        html.Div([
            html.Div(id="pron-holo", className="pron-holo"),
            html.Div([
                html.Span([html.I(className="pron-ley-barra"), "Ventas del trimestre"]),
                html.Span([html.I(className="pron-ley-barra pron"), "Pronóstico 2018 (±error típico)"]),
                html.Span([html.I(className="pron-ley-linea"), "Ventas de los últimos 12 meses · ◯ total del año"]),
            ], className="pron-leyenda"),
        ], className="pron-escena"),

        dcc.Store(id="pron-datos"),
        html.Div(id="pron-dummy", hidden=True),
    ], className="pron-vista")


@callback(Output("pron-datos", "data"), Output("pron-k17", "children"), Output("pron-kerr", "children"),
          Input("pron-cat", "value"), Input("pron-reg", "value"))
def preparar(cat, reg):
    d = datos(cat, reg)
    return d, f"${d['v17']:,.0f}", f"±{d['err']:.0f} %"


# La animación se dibuja en el navegador: al cambiar los datos o al presionar el botón
clientside_callback(
    "function (d, n) { return window.dash_clientside.pronostico.dibujar(d, n); }",
    Output("pron-dummy", "children"),
    Input("pron-datos", "data"), Input("pron-btn", "n_clicks"),
)
