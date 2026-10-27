"""tabs/conclusiones.py — Conclusiones del EDA y del modelo, recomendaciones y limitaciones."""
import dash_bootstrap_components as dbc
from dash import html

from common import load_artifact, modelo_listo, page_header, soft_card, usd


def _modelo():
    if not modelo_listo():
        return html.P("Entrene el modelo (python model/train_model.py) para ver sus conclusiones.", className="mb-0")
    r = load_artifact("resumen.json")
    p, rf, rec = r["prueba"], r["referencia"], r["prueba_recursiva"]
    return html.Ul([
        html.Li(f"{r['mejor_modelo']} fue el mejor modelo por validación cruzada. En 2017 —un año que no vio— "
                f"su error medio por serie fue de {usd(p['MAE'])}, un {(1 - p['MAE'] / rf['MAE']) * 100:.0f} % "
                f"menor que repetir el año anterior ({usd(rf['MAE'])})."),
        html.Li(f"Al proyectar 2017 completo solo con datos hasta 2016, el total anual quedó a "
                f"{rec['total_desvio_total_pct']:+.1f} % del real (la referencia, a "
                f"{rf['total_desvio_total_pct']:+.1f} %), con un error típico mensual de ±{rec['total_MAE_pct']:.0f} %."),
        html.Li("Las variables más importantes son el mes del año y las ventas del mismo mes del año anterior: "
                "el modelo confirma que la estacionalidad es el principal motor de las ventas."),
        html.Li(f"Para 2018 se proyectan {usd(r['proyeccion_2018'])}, "
                f"{(r['proyeccion_2018'] / r['ventas_2017'] - 1) * 100:+.0f} % frente a 2017, con el pico entre "
                f"septiembre y diciembre."),
    ], className="mb-0")


def layout():
    return html.Div([
        page_header("Conclusiones", "Lo que aprendimos, qué recomendamos y qué tener en cuenta"),
        dbc.Row([
            dbc.Col(soft_card("Análisis exploratorio", [
                html.P(["La distribución de ", html.Code("Sales"), " es muy sesgada hacia valores bajos, con "
                        "pocos pedidos de alto valor: los atípicos son el 11.7 % de los pedidos pero el 64 % de las "
                        "ventas, así que no se eliminan. ", html.Code("Category"), " es el factor que más explica "
                        "diferencias en el valor de venta: Technology y Furniture tienen pedidos más grandes, "
                        "mientras que Office Supplies domina en volumen con pedidos de menor valor."]),
                html.P([html.Code("Region"), " y ", html.Code("Segment"), " muestran diferencias leves. En el "
                        "tiempo, las ventas tienen un patrón estacional que se repite cada año (bajas en enero y "
                        "febrero, altas de septiembre a diciembre) y crecen desde 2016."], className="mb-0"),
            ], "blue"), lg=6, className="mb-3"),
            dbc.Col(soft_card("Modelo de proyección", _modelo(), "lavender"), lg=6, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(soft_card("Recomendaciones para el negocio", html.Ol([
                html.Li("Planificar inventario y personal para septiembre a diciembre a partir de la proyección "
                        "mensual, sobre todo en Technology y en las regiones West y East."),
                html.Li("Fijar las metas de cada región con la proyección como base y revisarlas cada mes con "
                        "las ventas reales: el error es menor un mes adelante que doce."),
                html.Li("Vigilar las combinaciones con caída proyectada (pestaña Proyección) para actuar con "
                        "anticipación."),
                html.Li("Reentrenar el modelo cada mes con los datos nuevos (python model/train_model.py)."),
            ], className="mb-0"), "peach"), lg=6, className="mb-3"),
            dbc.Col(soft_card("Limitaciones", html.Ul([
                html.Li("Dataset de ejemplo (2014–2017) de una sola tienda: no generaliza a otros negocios."),
                html.Li("Solo 36 meses aptos para modelar: las métricas tienen incertidumbre y un año atípico "
                        "las cambiaría."),
                html.Li("Sin precios, costos, campañas ni datos externos (economía, competencia): el modelo solo "
                        "conoce el historial de ventas y el calendario."),
                html.Li("La proyección de 2018 extiende el crecimiento reciente; si ese ritmo cambia, el error "
                        "crecerá con cada mes proyectado."),
                html.Li("A nivel de una categoría en una región el error mensual es alto; la proyección es más "
                        "confiable para totales (por región, por categoría o de la empresa)."),
                html.Li("El EDA es asociativo: no demuestra causalidad."),
            ], className="mb-0"), "green"), lg=6, className="mb-3"),
        ]),
    ])
