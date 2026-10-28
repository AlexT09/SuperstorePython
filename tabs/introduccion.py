"""tabs/introduccion.py — Presentación del proyecto, contexto e impacto para el negocio."""
import dash_bootstrap_components as dbc
from dash import html

from common import page_header, soft_card


def layout():
    return html.Div([
        page_header("Introducción", "Del análisis de las ventas de Sample Superstore a su proyección"),
        dbc.Row([
            dbc.Col(soft_card("El proyecto", [
                html.P("Sample Superstore es un conjunto de datos de muestra de una cadena minorista de "
                       "Estados Unidos (2014–2017) —una simulación con fines analíticos— con cerca de 10.000 "
                       "líneas de pedido de muebles, artículos de oficina y tecnología."),
                html.P("El proyecto tiene dos partes: un análisis exploratorio (EDA) que identifica qué "
                       "explica el nivel de ventas, y un modelo de machine learning que usa esos hallazgos "
                       "para proyectar cuánto venderá la empresa cada mes, por categoría y región.",
                       className="mb-0"),
            ], "blue"), md=7, className="mb-3"),
            dbc.Col(soft_card("Preguntas del proyecto", [
                html.P(["1. ¿Qué factores (categoría, región, segmento, tiempo) explican el nivel de ventas (",
                        html.Code("Sales"), ")?"], className="fw-bold"),
                html.P("2. ¿Cuánto venderá cada categoría en cada región, mes a mes, durante 2018?",
                       className="fw-bold"),
                html.P("Las pestañas siguen el flujo del proyecto: planteamiento, datos (ETL y EDA), "
                       "modelos y resultados, proyección y predicción, y conclusiones.", className="text-muted mb-0"),
            ], "lavender"), md=5, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(soft_card("Contexto", html.P(
                "Una cadena minorista vende a través de varias categorías de producto, regiones y "
                "segmentos de cliente, y sus ventas cambian a lo largo del año. Saber qué combinación de "
                "factores mueve las ventas, y anticipar cuánto se venderá, es un insumo directo para "
                "planificar: cuánto inventario comprar, qué metas fijar a cada región y cuándo reforzar "
                "la operación.", className="mb-0"), "peach"), md=6, className="mb-3"),
            dbc.Col(soft_card("Impacto para el negocio", html.Ul([
                html.Li("Metas comerciales: fijar la meta de cada región según lo que se espera vender."),
                html.Li("Inventario y compras: abastecer cada categoría antes de su temporada alta."),
                html.Li("Alerta temprana: detectar dónde se proyecta una caída frente al año anterior."),
                html.Li("Presupuesto: proyectar los ingresos del año siguiente."),
            ], className="mb-0"), "blue"), md=6, className="mb-3"),
        ]),
    ])
