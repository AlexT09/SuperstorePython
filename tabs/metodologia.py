"""tabs/metodologia.py — Flujo del proyecto, división temporal, validación y búsqueda de hiperparámetros."""
import dash_bootstrap_components as dbc
from dash import html

from common import page_header, soft_card

PASOS = [
    ("ETL", "CSV crudo → dataset limpio y tabla mensual (categoría × región × mes)"),
    ("EDA", "Distribución de las ventas, factores, estacionalidad y tendencia"),
    ("Variables", "Mes del año e historial de ventas, sin información del futuro"),
    ("Modelos", "7 regresores dentro de un Pipeline"),
    ("Ajuste", "Grid Search y Random Search con validación cruzada temporal"),
    ("Evaluación", "2017 reservado: MAE, RMSE y R² frente a la referencia"),
    ("Proyección", "Mejor modelo reentrenado con 2015–2017 → 2018 mes a mes"),
]


def layout():
    flujo = html.Div([
        html.Div([html.Span(str(i), className="flujo-num"), html.B(t), html.Span(d)], className="flujo-paso")
        for i, (t, d) in enumerate(PASOS, 1)], className="flujo")
    return html.Div([
        page_header("Metodología", "Cómo se pasa de los datos crudos a la proyección"),
        soft_card("Flujo del proyecto", flujo, "blue", height_full=False),
        dbc.Row([
            dbc.Col(soft_card("Datos y unidad de análisis", html.Ul([
                html.Li("Se agregan las 9.994 líneas de pedido en ventas mensuales por categoría × región: "
                        "12 series × 48 meses = 576 filas."),
                html.Li("2014 solo aporta historia (las variables necesitan 12 meses previos)."),
                html.Li(["Entrenamiento: ", html.B("2015–2016"), " (288 filas). Prueba: ", html.B("2017"),
                         " (144 filas), reservado y usado una sola vez."]),
                html.Li("La división es por fecha, no al azar: mezclar meses dejaría al modelo ver el futuro "
                        "(data leakage)."),
            ], className="mb-0"), "blue"), md=6, className="mt-3"),
            dbc.Col(soft_card("Variable objetivo", [
                html.P("El modelo predice un índice: cuántas veces el promedio de los últimos 12 meses se "
                       "venderá ese mes. Luego se multiplica por ese promedio para volver a dólares."),
                html.P("Así una misma regla sirve para series grandes (Technology en West) y pequeñas "
                       "(Furniture en South), y la proyección sigue el nivel reciente de cada serie. Con las "
                       "ventas en dólares directamente, la proyección anual se desviaba hasta un 38 %.",
                       className="mb-0"),
            ], "lavender"), md=6, className="mt-3"),
        ]),
        dbc.Row([
            dbc.Col(soft_card("Pipeline y validación", html.Ul([
                html.Li(["Pipeline de scikit-learn: ", html.Code("OneHotEncoder"), " (categoría, región) + ",
                         html.Code("StandardScaler"), " (numéricas) + modelo. El preprocesamiento se ajusta "
                         "solo con datos de entrenamiento en cada pliegue."]),
                html.Li([html.Code("TimeSeriesSplit"), " con 4 pliegues: cada uno entrena con los meses "
                         "anteriores y valida con los 3 meses siguientes."]),
                html.Li("Semilla fija (42) para que los resultados se puedan reproducir."),
            ], className="mb-0"), "peach"), md=6, className="mt-3"),
            dbc.Col(soft_card("Búsqueda de hiperparámetros", html.Ul([
                html.Li([html.B("Grid Search: "), "prueba todas las combinaciones de una rejilla definida."]),
                html.Li([html.B("Random Search: "), "prueba 12 combinaciones al azar de rangos continuos."]),
                html.Li("Criterio: menor MAE en validación cruzada. El mejor modelo se elige con ese error, "
                        "no con el de 2017, para que la prueba siga siendo independiente."),
                html.Li("Referencia: proyectar lo mismo que el mismo mes del año anterior. Un modelo solo "
                        "aporta si la supera."),
            ], className="mb-0"), "green"), md=6, className="mt-3"),
        ]),
    ])
