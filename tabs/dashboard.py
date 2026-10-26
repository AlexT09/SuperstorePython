"""tabs/dashboard.py — Panel de ventas embebido vía iframe.

El panel vive en panel/panel_ventas.html y app.py lo sirve en /panel/. Lee panel/panel_data.json,
que genera dashboard/build_dashboard.py con los datos históricos y los resultados del modelo
(app.py lo genera si no existe).
"""
from dash import html

from common import page_header, soft_card

PANEL_URL = "/panel/panel_ventas.html"


def layout():
    return html.Div([
        page_header("Presentación", "El proyecto completo en una sola página, para exponer: del "
                    "planteamiento a la proyección 2018 y las conclusiones"),
        soft_card("Panel de ventas · Sample Superstore", [
            html.Iframe(src=PANEL_URL, title="Panel de ventas",
                        style={"width": "100%", "height": "85vh", "border": "none", "borderRadius": "10px"}),
            html.A("Abrir en una pestaña nueva ↗", href=PANEL_URL, target="_blank",
                   className="small d-inline-block mt-2"),
        ], "lavender", height_full=False),
    ])
