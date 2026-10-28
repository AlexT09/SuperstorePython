"""tabs/proyeccion.py — Resultado final: proyección de ventas 2018 por categoría, región y mes."""
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dash_table, dcc, html

from tabs import prediccion
from common import (BRAND_LIGHT, NEG, POS, PROY, REAL, TEXT, aviso_sin_modelo, insight, kpi, load_artifact,
                    load_mensual, modelo_listo, page_header, soft_card, style_fig, usd)

CFG = {"displayModeBar": False}
TODAS = "Todas"
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def _filtrar(df, cat, reg):
    if cat != TODAS:
        df = df[df.Category == cat]
    if reg != TODAS:
        df = df[df.Region == reg]
    return df


def error_esperado(cat, reg):
    """Error de la proyección a 12 meses para este filtro, medido al proyectar 2017 con datos hasta 2016."""
    rec = _filtrar(load_artifact("prueba_recursiva_2017.csv"), cat, reg).groupby("Mes")[["Sales", "Proyeccion"]].sum()
    return float(np.mean(np.abs(rec.Sales - rec.Proyeccion)) / rec.Sales.mean() * 100)


def _tabla_series():
    m = load_mensual()
    real = m[m.Mes.dt.year == 2017].groupby(["Category", "Region"]).Sales.sum()
    proy = load_artifact("proyeccion_2018.csv").groupby(["Category", "Region"]).Proyeccion.sum()
    t = real.to_frame("v2017").join(proy.rename("v2018")).reset_index()
    t["crec"] = (t.v2018 / t.v2017 - 1) * 100
    t = t.sort_values("v2018", ascending=False)
    datos = [{"Categoría": f.Category, "Región": f.Region, "Ventas 2017": usd(f.v2017),
              "Proyección 2018": usd(f.v2018), "Variación": f"{f.crec:+.1f} %", "_c": f.crec} for f in t.itertuples()]
    return dash_table.DataTable(
        datos, [{"name": c, "id": c} for c in ["Categoría", "Región", "Ventas 2017", "Proyección 2018", "Variación"]],
        style_as_list_view=True, sort_action="native",
        style_cell={"fontFamily": "Inter, sans-serif", "padding": "7px", "color": TEXT, "textAlign": "left",
                    "backgroundColor": "rgba(0,0,0,0)", "border": "none",
                    "borderBottom": "1px solid rgba(255,255,255,.12)"},
        style_header={"backgroundColor": BRAND_LIGHT, "fontWeight": "600", "color": TEXT},
        style_data_conditional=[
            {"if": {"filter_query": "{_c} < 0", "column_id": "Variación"}, "color": NEG, "fontWeight": "700"},
            {"if": {"filter_query": "{_c} >= 15", "column_id": "Variación"}, "color": POS, "fontWeight": "700"},
        ],
        hidden_columns=["_c"], css=[{"selector": ".show-hide", "rule": "display: none"}],
    ), t


def layout():
    if not modelo_listo():
        return html.Div([page_header("Proyección 2018"), aviso_sin_modelo()])
    resumen = load_artifact("resumen.json")
    tabla, t = _tabla_series()
    caidas = t[t.crec < 0]
    top = t.sort_values("crec", ascending=False).iloc[0]
    p18 = load_artifact("proyeccion_2018.csv")
    temporada = p18[p18.Mes.dt.month >= 9].Proyeccion.sum() / p18.Proyeccion.sum() * 100
    sel = lambda id_, opts: dcc.Dropdown([TODAS] + opts, TODAS, id=id_, clearable=False)  # noqa: E731

    return html.Div([
        page_header("Proyección 2018", f"Ventas esperadas por mes con {resumen['mejor_modelo']}, reentrenado "
                    "con 2015–2017"),
        dbc.Row([
            dbc.Col([html.Label("Categoría", className="small fw-bold"),
                     sel("proy-cat", ["Furniture", "Office Supplies", "Technology"])], md=3),
            dbc.Col([html.Label("Región", className="small fw-bold"),
                     sel("proy-reg", ["Central", "East", "South", "West"])], md=3),
        ], className="mb-3"),
        dbc.Row(id="proy-kpis"),
        soft_card("Ventas mensuales: real 2015–2017 y proyección 2018", [
            dcc.Graph(id="proy-fig", config=CFG),
            insight("La banda muestra el error típico de una proyección a 12 meses para esta selección, medido "
                    "al proyectar 2017 sin conocerlo. Para el total de la empresa el error es menor que "
                    "para una sola categoría en una región, porque al sumar series los errores se compensan."),
        ], "blue", height_full=False),
        html.H4("Proyección por categoría y región", className="section-title"),
        dbc.Row([
            dbc.Col(soft_card("2018 frente a 2017", [
                tabla,
                insight("Ordenable por columna. En coral, las combinaciones donde se proyecta una caída; en "
                        "verde, crecimientos de 15 % o más."),
            ], "lavender"), lg=7, className="mb-3"),
            dbc.Col(soft_card("Lectura para el negocio", html.Ul([
                html.Li(f"Se proyectan {usd(resumen['proyeccion_2018'])} en 2018, "
                        f"{(resumen['proyeccion_2018'] / resumen['ventas_2017'] - 1) * 100:+.0f} % frente a 2017 "
                        f"({usd(resumen['ventas_2017'])}). El modelo extiende el crecimiento de 2016–2017."),
                html.Li(f"La temporada alta (septiembre a diciembre) concentra el {temporada:.0f} % de las ventas "
                        f"del año en solo 4 meses: es el periodo para asegurar inventario y personal."),
                html.Li(f"Mayor crecimiento esperado: {top.Category} en {top.Region} ({top.crec:+.0f} %)."),
                html.Li(f"Combinaciones con caída proyectada: {len(caidas)}"
                        + (": " + ", ".join(f"{c.Category} en {c.Region} ({c.crec:+.0f} %)" for c in caidas.itertuples())
                           if len(caidas) else ".")),
                html.Li("Usar la proyección como punto de partida para metas e inventario, ajustado con lo que "
                        "el modelo no conoce: campañas, precios y competencia."),
            ], className="mb-0"), "peach"), lg=5, className="mb-3"),
        ]),
        prediccion.simulador(),
    ])


@callback(Output("proy-fig", "figure"), Output("proy-kpis", "children"),
          Input("proy-cat", "value"), Input("proy-reg", "value"))
def actualizar(cat, reg):
    real = _filtrar(load_mensual(), cat, reg)
    real = real[real.Mes.dt.year >= 2015].groupby("Mes", as_index=False).Sales.sum()
    proy = _filtrar(load_artifact("proyeccion_2018.csv"), cat, reg).groupby("Mes", as_index=False).Proyeccion.sum()
    err = error_esperado(cat, reg)
    lo, hi = proy.Proyeccion * (1 - err / 100), proy.Proyeccion * (1 + err / 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(proy.Mes) + list(proy.Mes[::-1]), y=list(hi) + list(lo[::-1]), fill="toself",
                             fillcolor="rgba(240,184,81,0.16)", line_width=0, hoverinfo="skip",
                             name=f"Error típico (±{err:.0f} %)"))
    fig.add_trace(go.Scatter(x=real.Mes, y=real.Sales, name="Real", mode="lines+markers",
                             line=dict(color=REAL, width=2), marker_size=4))
    # La línea de la proyección arranca en el último mes real para que las dos curvas queden unidas
    p2 = pd.concat([real.tail(1).rename(columns={"Sales": "Proyeccion"}), proy])
    fig.add_trace(go.Scatter(x=p2.Mes, y=p2.Proyeccion, name="Proyección 2018", mode="lines+markers",
                             line=dict(color=PROY, width=2.5, dash="dash"), marker_size=5))
    fig.update_yaxes(title="Ventas del mes (USD)", tickprefix="$", tickformat=",.0f")
    fig = style_fig(fig, 380)

    v17 = real[real.Mes.dt.year == 2017].Sales.sum()
    v18 = proy.Proyeccion.sum()
    pico = proy.loc[proy.Proyeccion.idxmax()]
    kpis = [
        dbc.Col(kpi("Proyección 2018", usd(v18), f"real 2017: {usd(v17)}"), md=3, className="mb-3"),
        dbc.Col(kpi("Variación esperada", f"{(v18 / v17 - 1) * 100:+.1f} %", "frente a 2017"), md=3, className="mb-3"),
        dbc.Col(kpi("Mes pico", MESES[pico.Mes.month - 1].capitalize(), usd(pico.Proyeccion)), md=3, className="mb-3"),
        dbc.Col(kpi("Error típico mensual", f"±{err:.0f} %", "proyección a 12 meses (prueba 2017)"),
                md=3, className="mb-3"),
    ]
    return fig, kpis
