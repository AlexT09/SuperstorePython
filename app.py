"""
app.py — Proyección de ventas Sample Superstore (Dash + Bootstrap).
Diseño tipo panel de control: menú lateral fijo, barra superior y área de contenido.
Cada sección vive en tabs/<nombre>.py y expone una función layout().
Ejecutar:  python app.py   ->   http://127.0.0.1:8050
"""
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))  # clave y modelo de la IA (chatbot flotante)

from tabs import (asistente, conclusiones, eda, etl, introduccion, marco_teorico,  # noqa: E402
                  metodologia, modelos, problema, proyeccion)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
           suppress_callback_exceptions=True, title="Proyección de ventas Sample Superstore")
server = app.server


# Menú lateral agrupado: (grupo, [(id, etiqueta, icono de Bootstrap Icons, módulo), ...])
MENU = [
    ("Proyecto", [
        ("introduccion", "Introducción", "bi-house-door", introduccion),
        ("problema", "Problema y objetivos", "bi-bullseye", problema),
        ("marco", "Marco teórico", "bi-book", marco_teorico),
        ("metodologia", "Metodología", "bi-diagram-3", metodologia),
    ]),
    ("Datos", [
        ("etl", "ETL", "bi-database-gear", etl),
        ("eda", "EDA", "bi-bar-chart-line", eda),
    ]),
    ("Modelado", [
        ("modelos", "Modelos y resultados", "bi-cpu", modelos),
        ("proyeccion", "Proyección y predicción", "bi-graph-up-arrow", proyeccion),
    ]),
    ("Cierre", [
        ("conclusiones", "Conclusiones", "bi-check2-circle", conclusiones),
    ]),
]
INICIO = "introduccion"
SECCIONES = {tab_id: (grupo, label, mod) for grupo, items in MENU for tab_id, label, _, mod in items}


def ruta(tab_id: str) -> str:
    """La sección inicial vive en "/", el resto en "/<id>"."""
    return "/" if tab_id == INICIO else f"/{tab_id}"


def sidebar() -> html.Div:
    bloques = []
    for grupo, items in MENU:
        bloques.append(html.Div(grupo, className="sidebar-grupo"))
        bloques += [dbc.NavLink([html.I(className=f"bi {icono}", title=label), html.Span(label)],
                                href=ruta(tab_id), active="exact", className="sidebar-link")
                    for tab_id, label, icono, _ in items]
    return html.Div([
        html.Div([html.Div(html.I(className="bi bi-shop"), className="brand-logo"),
                  html.Div([html.Div("Superstore", className="brand-nombre"),
                            html.Div("Panel de ventas", className="brand-sub")])],
                 className="sidebar-brand"),
        dbc.Nav(bloques, vertical=True, pills=True, className="sidebar-nav"),
        html.Div([html.Div([html.Span("AT", className="avatar", style={"background": "#f0b851"}),
                            html.Span("DE", className="avatar", style={"background": "#5fc4b8"})],
                           className="avatares"),
                  html.Span("Alex Teran · David Estrada", className="nombres")], className="sidebar-pie"),
    ], className="sidebar")


app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Store(id="pestana-activa", data=INICIO),  # la lee el chatbot para dar sugerencias por sección
    sidebar(),
    html.Main([
        html.Header([
            html.Div([
                html.Button(html.I(className="bi bi-layout-sidebar"), className="btn-menu",
                            title="Plegar o desplegar el menú (tecla [)", **{"aria-label": "Plegar menú"}),
                html.Div([html.Div(id="topbar-ruta", className="topbar-ruta"),
                          html.H1("Proyección de ventas Sample Superstore", className="topbar-titulo")],
                         style={"minWidth": 0}),
            ], className="topbar-izq"),
            html.Div([
                html.Div([html.I(className="bi bi-calendar3"), "2014 – 2018"], className="topbar-chip mono"),
                html.Div([html.Span(className="topbar-punto"), "EDA + machine learning"], className="topbar-chip"),
            ], className="topbar-der"),
        ], className="topbar"),
        html.Div(dcc.Loading(html.Div(id="tab-content"), type="dot", color="#f0b851", delay_show=250),
                 className="contenido"),
    ], className="main"),
    asistente.widget(),  # chatbot flotante abajo a la derecha, visible en todas las secciones
], className="panel")


@app.callback(Output("tab-content", "children"), Output("topbar-ruta", "children"),
              Output("pestana-activa", "data"), Input("url", "pathname"))
def render_seccion(pathname):
    """Carga el layout() de la sección que indica la URL."""
    tab_id = (pathname or "/").strip("/") or INICIO
    if tab_id not in SECCIONES:
        tab_id = INICIO
    grupo, label, mod = SECCIONES[tab_id]
    migas = [html.Span(grupo), html.I(className="bi bi-chevron-right"), html.Span(label, className="actual")]
    return html.Div(mod.layout(), className="fade-in"), migas, tab_id


if __name__ == "__main__":
    app.run(debug=True)
