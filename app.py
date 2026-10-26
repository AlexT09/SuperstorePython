"""
app.py — Proyección de ventas Sample Superstore (Dash + Bootstrap).
Cada pestaña vive en tabs/<nombre>.py y expone una función layout().
Ejecutar:  python app.py   ->   http://127.0.0.1:8050
"""
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html
from flask import abort, send_from_directory

from common import modelo_listo
from dashboard.build_dashboard import DATA_OUT, PANEL_DIR, build
from tabs import (comparacion, conclusiones, dashboard, eda, etl, introduccion, marco_teorico, metodologia,
                  modelos, prediccion, problema, pronostico)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
           suppress_callback_exceptions=True, title="Proyección de ventas Sample Superstore")
server = app.server


# Panel de ventas (panel/). Se sirve aparte de assets/ para que Dash no cargue sus archivos en la app.
# Lee panel/panel_data.json, que se genera si falta (requiere el modelo entrenado).
@server.route("/panel/<path:filename>")
def panel(filename):
    if filename == DATA_OUT.name and not DATA_OUT.exists():
        if not modelo_listo():
            abort(404)
        build()
    return send_from_directory(PANEL_DIR, filename)


# (id, etiqueta, módulo) — el orden sigue el flujo del proyecto
TABS = [
    ("introduccion", "Introducción", introduccion),
    ("problema", "Problema y objetivos", problema),
    ("marco", "Marco teórico", marco_teorico),
    ("metodologia", "Metodología", metodologia),
    ("etl", "ETL", etl),
    ("eda", "EDA", eda),
    ("modelos", "Modelos y resultados", modelos),
    ("comparacion", "Comparación", comparacion),
    ("pronostico", "Proyección 2018", pronostico),
    ("prediccion", "Predicción", prediccion),
    ("conclusiones", "Conclusiones", conclusiones),
    ("dashboard", "Presentación", dashboard),
]
MODULES = {tab_id: mod for tab_id, _, mod in TABS}

app.layout = html.Div([
    html.Div(dbc.Container([
        html.H1("Proyección de ventas Sample Superstore"),
        html.P("Qué explica las ventas y cuánto se venderá · EDA y proyección con machine learning"),
    ]), className="app-header"),
    dbc.Container([
        dbc.Tabs([dbc.Tab(label=label, tab_id=tab_id) for tab_id, label, _ in TABS],
                 id="main-tabs", active_tab="introduccion", className="mb-4"),
        dcc.Loading(html.Div(id="tab-content"), type="dot", color="#2E6F95"),
        html.Div("Proyecto: Alex Teran y David Estrada", className="footer"),
    ], fluid="xl"),
])


@app.callback(Output("tab-content", "children"), Input("main-tabs", "active_tab"))
def render_tab(active_tab):
    """Carga el layout() del módulo correspondiente a la pestaña activa."""
    return html.Div(MODULES[active_tab].layout(), className="fade-in")


if __name__ == "__main__":
    app.run(debug=True)
