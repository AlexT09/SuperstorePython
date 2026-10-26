"""tabs/eda.py — Análisis exploratorio: univariado y bivariado."""
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html

from common import (ACCENT, BRAND_DARK, BRAND_DARKER, BRAND_LIGHT, CATEGORY_ACCENT, CATEGORY_COLORS,
                    COLORWAY, MARK, PASTEL, STATE_ABBR, TEXT, insight, load_data, page_header,
                    soft_card, style_fig)

CFG = {"displayModeBar": False}
MONO_SCALE = [[0, "#FFFFFF"], [0.5, PASTEL["yellow"]], [1, BRAND_DARKER]]


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
    "mapa": [
        "Los pedidos se concentran en pocos estados: California (2001), New York (1128) y Texas (985) "
        "suman el 41.2% del total, y son los estados con más pedidos de West, East y Central. Esto "
        "explica por qué West y East lideran en número de pedidos. En el otro extremo, varios estados "
        "tienen menos de 10 pedidos (p. ej. Wyoming con 1).",
    ],
    "box_categoria": [
        "Technology muestra el rango de ventas más alto, con algunos pedidos que superan los $20000. "
        "Office Supplies tiene la caja más baja, aunque algunos pedidos se acercan a $10000. Furniture es"
        " la categoría con el menor rango de ventas entre las tres.",
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
    "burbuja": [
        "Office Supplies tiene los círculos más grandes en todas las regiones, sobre todo en West y East."
        " Furniture tiene círculos intermedios, algo mayores en West y East. Technology presenta los "
        "círculos más pequeños en todas las regiones, coherente con ser la categoría de menor volumen de "
        "pedidos.",
    ],
    "bins2d": [
        "La mayoría de los pedidos se concentra en descuentos bajos (0.0-0.2), y las celdas con montos de"
        " venta más altos aparecen dispersas en varios niveles de descuento.",
    ],
    "correlacion": [
        "Sales y Profit tienen la correlación más alta del grupo (0.48, positiva moderada): a mayor "
        "venta, mayor ganancia, como es esperable. Sales y Quantity tienen una correlación débil (0.20). "
        "Sales y Discount tienen una correlación prácticamente nula (-0.03) en términos lineales, lo que "
        "contrasta con el patrón que sí se ve en el bins2D: la relación entre descuento y ventas no es "
        "lineal, por eso la correlación (que solo mide relación lineal) casi no la detecta.",
    ],
    "mensual": [
        "Los puntos muestran un patrón que se repite cada año: las ventas son más bajas a comienzos de "
        "año y suben hacia el final, lo que sugiere un comportamiento estacional.",
    ],
    "loess": [
        "Usamos la curva loess para visualizar la tendencia general. La línea de tendencia se mantiene "
        "casi plana entre 2014 y 2015 y sube con claridad desde 2016, lo que indica que las ventas "
        "totales crecieron en la segunda mitad del periodo, más que una tendencia lineal constante.",
    ],
    "tendencia_categoria": [
        "Las tres categorías crecen hacia el final del periodo, y Technology se mantiene casi siempre por"
        " encima de las otras dos. Esto es coherente con lo visto antes: Technology tiene menos pedidos "
        "pero de mayor valor. Technology presenta picos donde sobresale claramente por encima de las "
        "otras dos categorías, reforzando que es la categoría con mayor nivel de ventas.",
    ],
    "facetas_region": [
        "West tiene mayor estabilidad y niveles de venta ligeramente más altos en comparación con las "
        "demás a lo largo del tiempo.",
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
    return [go.Scatter(x=[c], y=[y], mode="markers", marker=dict(color=MARK, size=9, line=dict(width=1, color="white")),
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


def _mapa_estados(df):
    """Mapa coroplético de pedidos por estado (go.Choropleth con locationmode='USA-states')."""
    t = df.groupby(["State", "Region"]).size().reset_index(name="n")
    fig = go.Figure(go.Choropleth(
        locations=t.State.map(STATE_ABBR), z=t.n, locationmode="USA-states",
        colorscale=[[0, BRAND_LIGHT], [1, BRAND_DARKER]], colorbar_title="Pedidos",
        customdata=t[["State", "Region"]],
        hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>%{z:,} pedidos<extra></extra>",
        marker_line_color="white",
    ))
    fig.update_layout(geo=dict(scope="usa", bgcolor="rgba(0,0,0,0)"))
    style_fig(fig, 420)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    return fig


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
        dbc.Row([
            dbc.Col(soft_card("Mapa: pedidos por estado", [
                _graph(_mapa_estados(df)),
                _texto("mapa"),
            ], "blue"), md=12, className="mb-3"),
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
                               marker=dict(color=MARK, size=9, line=dict(width=1, color="white")), showlegend=False)
                    for c, y in d.groupby(col).log_sales.mean().items()])
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(title="Log(Ventas)")
    return style_fig(fig, 340)


def _burbuja(df):
    t = df.groupby(["Category", "Region"]).size().reset_index(name="n")
    fig = px.scatter(t, x="Category", y="Region", size="n", color="Category",
                     color_discrete_map=CATEGORY_COLORS, size_max=45)
    fig.update_traces(marker=dict(line=dict(width=1, color="white")))
    fig.update_xaxes(title="Categoría")
    fig.update_yaxes(title="Región")
    return style_fig(fig, 340)


def _bins2d(df):
    fig = px.density_heatmap(df, x="Discount", y="Sales", nbinsx=16, nbinsy=40,
                             color_continuous_scale=MONO_SCALE)
    fig.update_yaxes(range=[0, 1500])
    fig.update_layout(coloraxis_colorbar=dict(title="Frecuencia"))
    return style_fig(fig, 340)


def _corr(df):
    c = df[["Sales", "Quantity", "Discount", "Profit"]].corr().round(2)
    fig = px.imshow(c, text_auto=True, zmin=-1, zmax=1,
                    color_continuous_scale=[[0, "#FFFFFF"], [1, BRAND_DARK]])
    fig.update_layout(coloraxis_showscale=False)
    return style_fig(fig, 340)


def _tendencia_mensual(df):
    t = (df.assign(mes=df.Order_Date.dt.to_period("M").dt.to_timestamp())
           .groupby("mes", as_index=False).Sales.sum())
    fig = px.scatter(t, x="mes", y="Sales", color_discrete_sequence=[ACCENT["pink"]])
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color="white")))
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="Ventas")
    return style_fig(fig, 340), t


def _tendencia_loess(t):
    fig = px.scatter(t, x="mes", y="Sales", trendline="lowess", trendline_color_override=ACCENT["peach"],
                     color_discrete_sequence=[ACCENT["blue"]])
    fig.update_traces(selector=dict(mode="markers"), marker=dict(size=8, line=dict(width=1, color="white")))
    fig.update_traces(selector=dict(mode="lines"), line=dict(width=3))
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="Ventas")
    return style_fig(fig, 340)


def _tendencia_categoria(df):
    t = (df.assign(mes=df.Order_Date.dt.to_period("M").dt.to_timestamp())
           .groupby(["Category", "mes"], as_index=False).Sales.sum())
    fig = px.scatter(t, x="mes", y="Sales", color="Category", trendline="lowess",
                     color_discrete_map=CATEGORY_ACCENT)
    for trace in fig.data:
        if trace.mode == "markers":
            trace.showlegend = False
            trace.marker.update(size=6, opacity=.55)
        else:
            trace.line.update(width=3)
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="Ventas")
    return style_fig(fig, 340)


def _facetas_region(df):
    t = (df.assign(mes=df.Order_Date.dt.to_period("M").dt.to_timestamp())
           .groupby(["Region", "mes"], as_index=False).Sales.sum())
    fig = px.scatter(t, x="mes", y="Sales", facet_col="Region", facet_col_wrap=2,
                     trendline="lowess", trendline_color_override=ACCENT["peach"],
                     color_discrete_sequence=[ACCENT["blue"]])
    fig.update_traces(selector=dict(mode="markers"), marker=dict(size=6, line=dict(width=1, color="white")))
    fig.update_traces(selector=dict(mode="lines"), line=dict(width=3))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font=dict(color=TEXT, size=13)))
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="Ventas")
    return style_fig(fig, 460)


def _bivariado(df):
    return html.Div([
        html.H4("2 · Análisis bivariado", className="section-title"),
        dbc.Row([
            dbc.Col(soft_card("Boxplot: Sales por categoría", [
                _graph(_box_raw(df, "Category", "Categoría")),
                _texto("box_categoria"),
            ], "lavender"), lg=6, className="mb-3"),
            dbc.Col(soft_card("Boxplot: log(Sales) por categoría", [
                _graph(_box_log(df, "Category", "Categoría")),
                _texto("box_log_categoria"),
            ], "blue"), lg=6, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(soft_card("Boxplot: Sales por región", [
                _graph(_box_raw(df, "Region", "Región")),
                _texto("box_region"),
            ], "peach"), lg=6, className="mb-3"),
            dbc.Col(soft_card("Boxplot: Sales por segmento", [
                _graph(_box_raw(df, "Segment", "Segmento")),
                _texto("box_segmento"),
            ], "green"), lg=6, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(soft_card("Pedidos por categoría y región", [
                _graph(_burbuja(df)),
                _texto("burbuja"),
            ], "lavender"), lg=6, className="mb-3"),
            dbc.Col(soft_card("Bins 2D: Discount vs. Sales", [
                _graph(_bins2d(df)),
                _texto("bins2d"),
            ], "yellow"), lg=6, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(soft_card("Correlación entre variables continuas", [
                _graph(_corr(df)),
                _texto("correlacion"),
            ], "pink"), md=12, className="mb-3"),
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
            dbc.Col(soft_card("Tendencia de ventas en el tiempo (loess)", [
                _graph(_tendencia_loess(t_mensual)),
                _texto("loess"),
            ], "blue"), lg=6, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(soft_card("Tendencia de ventas por categoría", [
                _graph(_tendencia_categoria(df)),
                _texto("tendencia_categoria"),
            ], "lavender"), md=12, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(soft_card("Tendencia de ventas por región (facetas)", [
                _graph(_facetas_region(df)),
                _texto("facetas_region"),
            ], "green"), md=12, className="mb-3"),
        ]),
    ])


def layout():
    df = load_data()
    return html.Div([
        page_header("EDA", "Análisis univariado y bivariado: qué factores explican el nivel de ventas"),
        html.Div([_univariado(df), _bivariado(df), _tiempo(df)], className="fade-in"),
    ])
