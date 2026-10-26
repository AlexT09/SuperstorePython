"""tabs/problema.py — Planteamiento del problema y objetivos."""
import dash_bootstrap_components as dbc
from dash import html

from common import page_header, soft_card


def layout():
    return html.Div([
        page_header("Problema y objetivos", "Qué buscamos explicar, qué buscamos predecir y para qué"),
        dbc.Row([
            dbc.Col(soft_card("Planteamiento del problema", [
                html.P("Las ventas de Superstore varían según qué se vende (categoría), dónde (región), a "
                       "quién (segmento) y cuándo (mes y año). Sin entender esos factores, la empresa "
                       "planifica metas e inventario con base en intuición, y no puede anticipar los meses "
                       "de mayor o menor demanda."),
                html.P(["Pregunta 1 (EDA): ¿qué factores explican el nivel de ventas (", html.Code("Sales"),
                        ")?"], className="fw-bold mb-1"),
                html.P("Pregunta 2 (modelo): ¿cuánto venderá cada categoría en cada región, mes a mes, "
                       "durante 2018?", className="fw-bold mb-0"),
            ], "lavender"), md=12, className="mb-3"),
            dbc.Col(soft_card("Objetivo general", html.P(
                "Identificar los factores que explican las ventas de Sample Superstore y construir un "
                "modelo de machine learning que proyecte las ventas mensuales por categoría y región "
                "para apoyar la planificación comercial.", className="fs-5 mb-0"), "pink"),
                md=12, className="mb-3"),
            dbc.Col(soft_card("Objetivos específicos", html.Ol([
                html.Li(["Preparar los datos (ETL) y construir la tabla de ventas mensuales por categoría × "
                         "región."]),
                html.Li(["Analizar la distribución de ", html.Code("Sales"), " y cómo varía según ",
                         html.Code("Category"), ", ", html.Code("Region"), ", ", html.Code("Segment"),
                         " y el tiempo (EDA)."]),
                html.Li("Entrenar y ajustar 7 modelos de regresión (Ridge, Lasso, KNN, Árbol de "
                        "decisión, Random Forest, XGBoost y SVR)."),
                html.Li("Compararlos entre sí y contra una referencia simple (vender lo mismo que el mismo "
                        "mes del año anterior) con MAE, RMSE y R²."),
                html.Li("Proyectar las ventas de 2018 con el mejor modelo y traducirlas en "
                        "recomendaciones para el negocio."),
            ], className="mb-0"), "blue"), md=7, className="mb-3"),
            dbc.Col(soft_card("Justificación", html.P(
                "Las ventas tienen un patrón estacional claro (suben hacia fin de año) y una tendencia de "
                "crecimiento desde 2016, como muestra el EDA. Ese comportamiento repetible es lo que hace "
                "posible proyectarlas. Se proyecta por categoría y región porque es el nivel al que "
                "se fijan metas e inventario, y porque a nivel de producto o de día los datos son "
                "demasiado escasos y ruidosos.", className="mb-0"), "peach"), md=5, className="mb-3"),
        ]),
    ])
