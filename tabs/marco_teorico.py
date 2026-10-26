"""tabs/marco_teorico.py — Conceptos, modelos del curso y tabla de operacionalización de variables."""
import dash_bootstrap_components as dbc
from dash import html

from common import page_header, soft_card

# (variable, definición conceptual, tipo, escala / valores, rol)
VARIABLES_DATASET = [
    ("Sales", "Valor monetario de la línea de pedido", "Numérica continua", "USD (≥ 0)", "EDA"),
    ("log(Sales)", "Logaritmo natural de Sales (corrige la asimetría)", "Numérica continua", "Reales", "EDA"),
    ("Category", "Familia de producto", "Categórica nominal", "Furniture, Office Supplies, Technology", "EDA y modelo"),
    ("Sub-Category", "Subcategoría de producto dentro de Category", "Categórica nominal", "17 subcategorías (p. ej. Chairs, Binders, Phones)", "Descriptiva (no se usa)"),
    ("Region", "Región geográfica del pedido", "Categórica nominal", "Central, East, South, West", "EDA y modelo"),
    ("Segment", "Tipo de cliente", "Categórica nominal", "Consumer, Corporate, Home Office", "EDA"),
    ("Ship Mode", "Modalidad de envío", "Categórica nominal", "Standard, Second, First Class, Same Day", "Descriptiva (no se usa)"),
    ("Quantity", "Unidades por línea de pedido", "Numérica discreta", "Enteros (1–14)", "Descriptiva (no se usa)"),
    ("Discount", "Descuento aplicado", "Numérica continua", "Proporción 0–0.8", "Descriptiva (no se usa)"),
    ("Profit", "Ganancia de la línea de pedido", "Numérica continua", "USD (puede ser negativa)", "Descriptiva (no se usa)"),
    ("Order Date", "Fecha del pedido", "Fecha", "2014-01-03 a 2017-12-30", "EDA y modelo (mes)"),
]

VARIABLES_MODELO = [
    ("Ventas del mes", "Suma de Sales de una categoría en una región durante un mes", "Numérica continua", "USD", "Lo que se proyecta"),
    ("Índice de ventas", "Ventas del mes ÷ promedio de los 12 meses anteriores (1.3 = 30 % más que el promedio)", "Numérica continua", "≥ 0", "Variable objetivo (y)"),
    ("Categoría", "Familia de producto de la serie", "Categórica nominal (one-hot)", "3 valores", "Variable independiente"),
    ("Región", "Región de la serie", "Categórica nominal (one-hot)", "4 valores", "Variable independiente"),
    ("Mes del año", "Posición del mes en el año como punto en un círculo (seno y coseno)", "Numérica continua", "−1 a 1", "Variable independiente (estacionalidad)"),
    ("Ventas del mismo mes del año anterior", "Ventas de hace 12 meses ÷ promedio de 12 meses", "Numérica continua", "≥ 0", "Variable independiente"),
    ("Ventas del mes anterior", "Ventas de hace 1 mes ÷ promedio de 12 meses", "Numérica continua", "≥ 0", "Variable independiente"),
    ("Promedio de los últimos 3 meses", "Media de los 3 meses previos ÷ promedio de 12 meses", "Numérica continua", "≥ 0", "Variable independiente (ritmo reciente)"),
]

MODELOS = [
    ("Ridge y Lasso", "Regresión lineal con penalización (L2 y L1) que reduce el sobreajuste; Lasso puede anular variables poco útiles."),
    ("KNN", "Promedia las ventas de los k meses más parecidos del historial."),
    ("Árbol de decisión", "Divide los datos con reglas sí/no (p. ej. «¿es noviembre?») hasta agrupar meses similares."),
    ("Random Forest", "Promedia cientos de árboles entrenados con muestras distintas: más estable que un solo árbol."),
    ("XGBoost", "Encadena árboles pequeños donde cada uno corrige el error de los anteriores (boosting)."),
    ("SVR", "Busca una función que deje la mayoría de los puntos dentro de un margen de error ε, con kernels para relaciones no lineales."),
]


def _tabla(filas, encabezados):
    return dbc.Table(
        [html.Thead(html.Tr([html.Th(h) for h in encabezados])),
         html.Tbody([html.Tr([html.Td(html.B(v[0]))] + [html.Td(c) for c in v[1:]]) for v in filas])],
        hover=True, responsive=True, borderless=True, className="mb-0",
    )


def layout():
    enc = ["Variable", "Definición", "Tipo", "Escala / valores", "Rol"]
    return html.Div([
        page_header("Marco teórico", "Conceptos clave, modelos y operacionalización de variables"),
        dbc.Row([
            dbc.Col(soft_card("Análisis exploratorio (EDA)", html.P(
                "Resume y visualiza los datos antes de modelar: variación de cada variable (univariado) y "
                "covariación entre variables (bivariado). Sus hallazgos orientan qué variables entran al "
                "modelo.", className="mb-0"), "blue"), md=4, className="mb-3"),
            dbc.Col(soft_card("Regresión supervisada", html.P(
                "Aprende una función que relaciona variables de entrada (X) con un valor numérico (y) a "
                "partir de ejemplos conocidos. Aquí y son las ventas de un mes y X lo que se sabe antes de "
                "que empiece: categoría, región, mes del año e historial de ventas.", className="mb-0"),
                "lavender"), md=4, className="mb-3"),
            dbc.Col(soft_card("Generalización y sobreajuste", html.P(
                "Un modelo sirve si acierta en datos que no vio. Por eso se ajusta con validación cruzada y "
                "se evalúa una sola vez en un periodo reservado (2017). Con datos en el tiempo, el modelo "
                "nunca debe entrenar con meses posteriores a los que valida.", className="mb-0"), "peach"),
                md=4, className="mb-3"),
        ]),
        html.H4("Modelos de regresión utilizados", className="section-title"),
        soft_card("Modelos", _tabla(MODELOS, ["Modelo", "Idea principal"]), "blue"),
        html.H4("Métricas de evaluación", className="section-title"),
        dbc.Row([
            dbc.Col(soft_card("MAE", html.P("Error absoluto medio: cuántos dólares se equivoca la proyección "
                                            "en promedio. Es la métrica principal porque se lee en dólares.",
                                            className="mb-0"), "blue"), md=4, className="mb-3"),
            dbc.Col(soft_card("RMSE", html.P("Raíz del error cuadrático medio: castiga más los errores grandes "
                                             "(un mes muy mal proyectado pesa más).", className="mb-0"),
                              "lavender"), md=4, className="mb-3"),
            dbc.Col(soft_card("R²", html.P("Proporción de la variación de las ventas que el modelo explica "
                                           "(1 = perfecto, 0 = igual que usar el promedio).", className="mb-0"),
                              "peach"), md=4, className="mb-3"),
        ]),
        html.H4("Operacionalización de variables", className="section-title"),
        soft_card("Variables del dataset", _tabla(VARIABLES_DATASET, enc), "green", height_full=False),
        html.Div(className="mb-3"),
        soft_card("Variables del modelo (una fila = categoría × región × mes)", _tabla(VARIABLES_MODELO, enc),
                  "green", height_full=False),
    ])
