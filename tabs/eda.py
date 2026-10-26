"""tabs/eda.py — Análisis exploratorio: univariado, bivariado y ventas en el tiempo."""
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html

from common import (ACCENT, CATEGORY_COLORS, COLORWAY, MARK, PASTEL, PROY, SEQ_SCALE, insight, load_data,
                    page_header, soft_card, style_fig)

CFG = {"displayModeBar": False}
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


# Interpretación de cada gráfico. Cada elemento de la lista es un párrafo.
TEXTOS = {
    "resumen": [
        "La media (230) es 4 veces la mediana (54.5) y la desviación estándar (623) supera a la media, lo"
        " que confirma una fuerte asimetría positiva: la mayoría de los pedidos son de bajo valor y unos "
        "pocos pedidos muy grandes (hasta 22638) jalan el promedio hacia arriba. El RIC de 193 (entre "
        "Q1=17.3 y Q3=210) muestra que el 50% central de los pedidos se mueve en un rango moderado.",
    ],
    "histograma": [
        "Confirma visualmente el sesgo a la derecha: la gran mayoría de las barras se concentran en "
        "valores bajos de venta, con una cola larga y dispersa hacia la derecha.",
    ],
    "densidad_log": [
        "Al aplicar el logaritmo, la distribución se vuelve más simétrica y manejable. La mayor "
        "concentración de observaciones se encuentra aproximadamente entre log(Sales) = 2.5 y 4, donde "
        "está el máximo de la densidad. Hacia la derecha la densidad disminuye, con un pequeño "
        "abultamiento cerca de 5, mostrando una menor concentración de ventas altas.",
    ],
    "barras_categoria": [
        "Office Supplies concentra el 60.3% de los pedidos (6026), muy por encima de Furniture (21.2%, "
        "2121) y Technology (18.5%, 1847). Es decir, Office Supplies es la categoría que más rota en "
        "número de pedidos.",
    ],
    "barras_region": [
        "West tiene mayor numero de pedidos con 3203 pedidos, seguida de East con 2848. Central con 2323 "
        "y South con menor numero de pedidos con 1620.",
    ],
    "box_log_categoria": [
        "Furniture (5.205) con la mediana mas alta junto con Technology (5.113). Office Supplies queda "
        "muy por debajo (3.311), casi 2 puntos log menos que las otras dos.",
        "En el ancho de la caja (RIC), Technology (1.886) y Office Supplies (1.916) son muy parecidas, y "
        "Furniture es la que tiene la caja más ancha/dispersa (2.225).",
    ],
    "box_region": [
        "Las cajas de las cuatro regiones son casi iguales; lo que cambia son los valores atípicos, con "
        "pedidos puntuales muy grandes en South y Central. A diferencia de Category, las diferencias "
        "entre regiones son menos marcadas.",
    ],
    "box_segmento": [
        "Medianas: Consumer 53.72, Corporate 56.54, Home Office 52.44, ningun segmento sobresale.",
        "Medias: Consumer 223.73, Corporate 233.82, Home Office 240.9, aquí Home Office tiene la media "
        "más alta.",
        "Consumer tiene 5191 pedidos, casi el triple que Home Office (1783). Consumer genera más ventas "
        "totales por tener más pedidos.",
    ],
    "mensual": [
        "Los puntos muestran un patrón que se repite cada año: las ventas son más bajas a comienzos de "
        "año y suben hacia el final, lo que sugiere un comportamiento estacional.",
    ],
    "tendencia": [
        "El promedio móvil de 12 meses quita la estacionalidad y deja ver la tendencia: se mantiene casi "
        "plano, en torno a $40.000 por mes, entre 2014 y 2015 y sube con claridad desde 2016, hasta unos "
        "$61.000 a fines de 2017. Las ventas crecieron en la segunda mitad del periodo, más que con una "
        "tendencia lineal constante.",
    ],
    "estacionalidad": [
        "Cada fila suma el 100 % de las ventas del año. El patrón se repite los cuatro años: enero y "
        "febrero son los meses más bajos (entre 1 % y 6 % del año cada uno) y septiembre, noviembre y "
        "diciembre los más altos (entre 11 % y 17 %). Por eso el modelo usa el mes del año y las ventas "
        "del mismo mes del año anterior.",
    ],
    "tendencia_categoria": [
        "Las líneas son el promedio móvil de 12 meses de cada categoría. Las tres crecen hacia el final "
        "del periodo, y Technology tiene el nivel más alto en 27 de los 37 meses, coherente con lo visto "
        "antes: menos pedidos pero de mayor valor. Office Supplies es la que más acelera en 2017.",
    ],
}


def _texto(key):
    """Recuadro de interpretación con los párrafos de TEXTOS[key] (separados por saltos de línea)."""
    parts = TEXTOS[key]
    children = [x for i, p in enumerate(parts) for x in ([html.Br()] if i else []) + [p]]
    return insight(children if len(parts) > 1 else parts[0])

def _graph(fig, height=None):
    return dcc.Graph(figure=fig if height is None else style_fig(fig, height), config=CFG)


def _mean_markers(df, col):
    """Punto sobre la media de cada grupo (equivalente a stat_summary de ggplot)."""
    return [go.Scatter(x=[c], y=[y], mode="markers", marker=dict(color=MARK, size=9, line=dict(width=1, color="rgba(4,18,27,.85)")),
                       showlegend=False) for c, y in df.groupby(col).Sales.mean().items()]


# ------------------------------------------------------------ análisis univariado
def _resumen_tabla(df):
    s = df.Sales
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    filas = [
        ("n", f"{len(s):,}"), ("media", f"{s.mean():.0f}"), ("ds", f"{s.std():.0f}"),
        ("mediana", f"{s.median():.1f}"), ("q1", f"{q1:.1f}"), ("q3", f"{q3:.0f}"),
        ("RIC", f"{q3 - q1:.0f}"), ("minimo", f"{s.min():.1f}"), ("maximo", f"{s.max():.0f}"),
    ]
    return dbc.Table(
        [html.Thead(html.Tr([html.Th(k) for k, _ in filas])), html.Tbody(html.Tr([html.Td(v) for _, v in filas]))],
        hover=True, responsive=True, borderless=True, size="sm",
    )


def _hist_sales(df, color):
    fig = px.histogram(df[df.Sales <= 1000], x="Sales", nbins=40, color_discrete_sequence=[PASTEL[color]])
    fig.update_traces(marker_line_width=0)
    fig.update_xaxes(title="Ventas")
    fig.update_yaxes(title="Número de pedidos", title_standoff=12, automargin=True)
    return style_fig(fig, 320)


def _hist_log(df, color):
    fig = px.histogram(df, x=np.log(df.Sales), nbins=50, histnorm="probability density",
                       color_discrete_sequence=[PASTEL[color]])
    fig.update_traces(marker_line_width=0)
    fig.update_xaxes(title="Log(Ventas)")
    fig.update_yaxes(title="Densidad", title_standoff=12, automargin=True)
    return style_fig(fig, 320)


def _barras(df, col, xlabel, color):
    t = df[col].value_counts().rename_axis(col).reset_index(name="n")
    t["pct"] = t.n / t.n.sum() * 100
    t = t.sort_values("n")
    fig = px.bar(t, x=col, y="n", text=t.apply(lambda r: f"{r.n:,} ({r.pct:.1f}%)", axis=1),
                 color_discrete_sequence=[PASTEL[color]])
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(title="Número de pedidos")
    return style_fig(fig, 320)


def _univariado(df):
    return html.Div([
        html.H4("1 · Análisis univariado", className="section-title"),
        dbc.Row([
            dbc.Col(soft_card("Resumen estadístico de Sales", [
                _resumen_tabla(df),
                _texto("resumen"),
            ], "blue"), md=12, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(soft_card("Histograma de Sales", [
                _graph(_hist_sales(df, "peach")),
                _texto("histograma"),
            ], "peach"), lg=6, className="mb-3"),
            dbc.Col(soft_card("Densidad de log(Sales)", [
                _graph(_hist_log(df, "pink")),
                _texto("densidad_log"),
            ], "pink"), lg=6, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(soft_card("Barras: pedidos por categoría", [
                _graph(_barras(df, "Category", "Categoría", "blue")),
                _texto("barras_categoria"),
            ], "blue"), lg=6, className="mb-3"),
            dbc.Col(soft_card("Barras: pedidos por región", [
                _graph(_barras(df, "Region", "Región", "green")),
                _texto("barras_region"),
            ], "green"), lg=6, className="mb-3"),
        ]),
    ])


# ------------------------------------------------------------- análisis bivariado
def _box_raw(df, col, xlabel):
    seq = None if col == "Category" else COLORWAY
    cmap = CATEGORY_COLORS if col == "Category" else None
    fig = px.box(df, x=col, y="Sales", color=col, color_discrete_map=cmap, color_discrete_sequence=seq)
    fig.add_traces(_mean_markers(df, col))
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(title="Ventas")
    return style_fig(fig, 340)


def _box_log(df, col, xlabel):
    d = df.assign(log_sales=np.log(df.Sales))
    seq = None if col == "Category" else COLORWAY
    cmap = CATEGORY_COLORS if col == "Category" else None
    fig = px.box(d, x=col, y="log_sales", color=col, color_discrete_map=cmap, color_discrete_sequence=seq)
    fig.add_traces([go.Scatter(x=[c], y=[y], mode="markers",
                               marker=dict(color=MARK, size=9, line=dict(width=1, color="rgba(4,18,27,.85)")), showlegend=False)
                    for c, y in d.groupby(col).log_sales.mean().items()])
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(title="Log(Ventas)")
    return style_fig(fig, 340)


def _tendencia_mensual(df):
    t = (df.assign(mes=df.Order_Date.dt.to_period("M").dt.to_timestamp())
           .groupby("mes", as_index=False).Sales.sum())
    fig = px.scatter(t, x="mes", y="Sales", color_discrete_sequence=[ACCENT["pink"]])
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color="rgba(4,18,27,.85)")))
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="Ventas")
    return style_fig(fig, 340), t


def _tendencia_movil(t):
    """Ventas mensuales con su promedio móvil de 12 meses (quita la estacionalidad)."""
    t = t.assign(movil=t.Sales.rolling(12).mean())
    fig = go.Figure([
        go.Scatter(x=t.mes, y=t.Sales, mode="markers", name="Ventas del mes",
                   marker=dict(size=8, color=ACCENT["blue"], line=dict(width=1, color="rgba(4,18,27,.85)"))),
        go.Scatter(x=t.mes, y=t.movil, mode="lines", name="Promedio móvil de 12 meses",
                   line=dict(width=3, color=PROY)),
    ])
    fig.update_yaxes(title="Ventas", tickprefix="$", tickformat=",.0f")
    return style_fig(fig, 340)


def _estacionalidad(df):
    """Mapa de calor mes × año: porcentaje de las ventas del año que cae en cada mes."""
    t = df.assign(anio=df.Order_Date.dt.year, mes=df.Order_Date.dt.month).pivot_table(
        index="anio", columns="mes", values="Sales", aggfunc="sum")
    pct = t.div(t.sum(axis=1), axis=0) * 100
    fig = go.Figure(go.Heatmap(z=pct.values, x=MESES, y=[str(a) for a in pct.index], colorscale=SEQ_SCALE,
                               text=pct.round(1).values, texttemplate="%{text}", xgap=3, ygap=3,
                               hovertemplate="%{y} · %{x}: %{z:.1f} % del año<extra></extra>",
                               colorbar=dict(title="% del año")))
    fig.update_yaxes(autorange="reversed")
    return style_fig(fig, 340)


def _tendencia_categoria(df):
    t = (df.assign(mes=df.Order_Date.dt.to_period("M").dt.to_timestamp())
           .groupby(["Category", "mes"]).Sales.sum().unstack(0))
    fig = go.Figure()
    for cat in t.columns:
        color = CATEGORY_COLORS[cat]
        fig.add_trace(go.Scatter(x=t.index, y=t[cat], mode="markers", marker=dict(size=6, color=color, opacity=.5),
                                 showlegend=False, hovertemplate="%{y:$,.0f}<extra>" + cat + "</extra>"))
        fig.add_trace(go.Scatter(x=t.index, y=t[cat].rolling(12).mean(), mode="lines", name=cat,
                                 line=dict(width=3, color=color),
                                 hovertemplate="%{y:$,.0f}<extra>" + cat + " (promedio móvil)</extra>"))
    fig.update_yaxes(title="Ventas", tickprefix="$", tickformat=",.0f")
    return style_fig(fig, 340)


def _bivariado(df):
    return html.Div([
        html.H4("2 · Análisis bivariado", className="section-title"),
        dbc.Row([
            dbc.Col(soft_card("Boxplot: log(Sales) por categoría", [
                _graph(_box_log(df, "Category", "Categoría")),
                _texto("box_log_categoria"),
            ], "lavender"), lg=4, className="mb-3"),
            dbc.Col(soft_card("Boxplot: Sales por región", [
                _graph(_box_raw(df, "Region", "Región")),
                _texto("box_region"),
            ], "peach"), lg=4, className="mb-3"),
            dbc.Col(soft_card("Boxplot: Sales por segmento", [
                _graph(_box_raw(df, "Segment", "Segmento")),
                _texto("box_segmento"),
            ], "green"), lg=4, className="mb-3"),
        ]),
    ])


def _tiempo(df):
    fig_mensual, t_mensual = _tendencia_mensual(df)
    return html.Div([
        html.H4("3 · Ventas en el tiempo", className="section-title"),
        dbc.Row([
            dbc.Col(soft_card("Evolución mensual de ventas", [
                _graph(fig_mensual),
                _texto("mensual"),
            ], "pink"), lg=6, className="mb-3"),
            dbc.Col(soft_card("Tendencia: promedio móvil de 12 meses", [
                _graph(_tendencia_movil(t_mensual)),
                _texto("tendencia"),
            ], "blue"), lg=6, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(soft_card("Estacionalidad: % de las ventas del año por mes", [
                _graph(_estacionalidad(df)),
                _texto("estacionalidad"),
            ], "yellow"), lg=6, className="mb-3"),
            dbc.Col(soft_card("Tendencia de ventas por categoría", [
                _graph(_tendencia_categoria(df)),
                _texto("tendencia_categoria"),
            ], "lavender"), lg=6, className="mb-3"),
        ]),
    ])


def layout():
    df = load_data()
    return html.Div([
        page_header("EDA", "Análisis univariado, bivariado y en el tiempo: qué factores explican el nivel de ventas"),
        html.Div([_univariado(df), _bivariado(df), _tiempo(df)], className="fade-in"),
    ])
