"""tabs/eda.py — Análisis exploratorio: univariado, bivariado y ventas en el tiempo."""
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html

from common import (ACCENT, CATEGORY_COLORS, COLORWAY, DIV_SCALE, MARK, NEG, PASTEL, POS, PROY, REAL, SEQ_SCALE,
                    insight, load_data,
                    load_mensual, page_header, soft_card, style_fig)
from model.features import agregar_variables

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


# ------------------------------------------------- atípicos y relación con el modelo
def _atipicos(df):
    """Regla del rango intercuantil: atípico si supera Q3 + 1.5 × RIC. Compara su peso en pedidos y en ventas."""
    s = df.Sales
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    lim = q3 + 1.5 * (q3 - q1)
    out = s > lim
    pct_ped, pct_ven = out.mean() * 100, s[out].sum() / s.sum() * 100
    grupos = ["Pedidos", "Ventas (USD)"]
    fig = go.Figure([
        go.Bar(x=grupos, y=[100 - pct_ped, 100 - pct_ven], name=f"Normales (≤ ${lim:,.0f})", marker_color=REAL,
               text=[f"{100 - pct_ped:.1f} %", f"{100 - pct_ven:.1f} %"], textposition="inside"),
        go.Bar(x=grupos, y=[pct_ped, pct_ven], name=f"Atípicos (> ${lim:,.0f})", marker_color=PROY,
               text=[f"{pct_ped:.1f} %", f"{pct_ven:.1f} %"], textposition="inside", textfont=dict(color="#04121b")),
    ])
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="% del total", ticksuffix=" %", range=[0, 100])
    texto = (f"Con la regla del rango intercuantil (atípico si supera Q3 + 1.5 × RIC = ${lim:,.0f}), "
             f"{out.sum():,} pedidos son atípicos: el {pct_ped:.1f} % de los pedidos, pero el {pct_ven:.0f} % de "
             f"las ventas. No son errores de captura sino los pedidos grandes que sostienen el negocio, por eso "
             f"el ETL no los elimina. Para el modelo, estos pedidos explican los picos de algunos meses que "
             f"ninguna variable puede anticipar, y por eso se trabaja con ventas mensuales y no por pedido.")
    return style_fig(fig, 340), texto


def _dispersion_rezago():
    """Índice del mes vs. índice del mismo mes del año anterior, con recta de mínimos cuadrados."""
    m = agregar_variables(load_mensual()).dropna(subset=["indice", "r_lag_12", "r_lag_1"])
    r12, r1 = m.indice.corr(m.r_lag_12), m.indice.corr(m.r_lag_1)
    pend, inter = np.polyfit(m.r_lag_12, m.indice, 1)
    xs = np.array([m.r_lag_12.min(), m.r_lag_12.max()])
    fig = go.Figure([go.Scatter(x=g.r_lag_12, y=g.indice, mode="markers", name=cat,
                                marker=dict(size=7, color=CATEGORY_COLORS[cat], opacity=.75,
                                            line=dict(width=1, color="rgba(4,18,27,.85)")),
                                hovertemplate="%{x:.2f} → %{y:.2f}<extra>" + cat + "</extra>")
                     for cat, g in m.groupby("Category")])
    fig.add_trace(go.Scatter(x=xs, y=inter + pend * xs, mode="lines", name="Tendencia (mínimos cuadrados)",
                             line=dict(color=PROY, width=3)))
    fig.update_xaxes(title="Índice del mismo mes del año anterior")
    fig.update_yaxes(title="Índice del mes")
    texto = (f"Cada punto es una categoría en una región durante un mes (2015–2017), en índice: ventas del mes "
             f"÷ promedio de los 12 meses anteriores. La línea sube: un mes que vendió por encima de lo normal "
             f"el año pasado tiende a repetirlo (correlación {r12:.2f}). Con el mes anterior la correlación es "
             f"casi nula ({r1:.2f}). Es la relación que usa el modelo, y anticipa lo que muestra su importancia "
             f"de variables: pesa el mismo mes del año anterior, no el mes anterior. La nube es ancha, por eso "
             f"el error por serie es alto.")
    return style_fig(fig, 340), texto


def _atipicos_modelo(df):
    fig_a, txt_a = _atipicos(df)
    fig_d, txt_d = _dispersion_rezago()
    return html.Div([
        html.H4("4 · Atípicos y relación con el modelo", className="section-title"),
        dbc.Row([
            dbc.Col(soft_card("Atípicos: su peso en pedidos y en ventas", [
                _graph(fig_a), insight(txt_a),
            ], "yellow"), lg=6, className="mb-3"),
            dbc.Col(soft_card("Dispersión: el mes contra el mismo mes del año anterior", [
                _graph(fig_d), insight(txt_d),
            ], "blue"), lg=6, className="mb-3"),
        ]),
    ])


# ------------------------------------------------------------ geografía (mapas)
ABREV = {
    "Alabama": "AL", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO", "Connecticut": "CT",
    "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
    "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
    "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}


def _por_estado(df):
    st = df.groupby("State").agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), pedidos=("Sales", "size"))
    st["margen"] = st.Profit / st.Sales * 100
    st["share"] = st.Sales / st.Sales.sum() * 100
    st["code"] = st.index.map(ABREV)
    return st.sort_values("Sales", ascending=False)


def _mapa(st, col, escala, titulo_barra, formato, zmid=None):
    fig = go.Figure(go.Choropleth(
        locations=st.code, z=st[col], locationmode="USA-states", colorscale=escala, zmid=zmid,
        marker_line_color="rgba(4,18,27,.6)", marker_line_width=.6, text=st.index,
        customdata=np.stack([st.Sales, st.margen, st.pedidos], axis=1),
        hovertemplate="<b>%{text}</b><br>Ventas: $%{customdata[0]:,.0f}<br>Margen: %{customdata[1]:.1f} %"
                      "<br>Pedidos: %{customdata[2]:,}<extra></extra>",
        colorbar=dict(title=titulo_barra, tickformat=formato, thickness=12, len=.8),
    ))
    fig.update_geos(scope="usa", bgcolor="rgba(0,0,0,0)", lakecolor="rgba(0,0,0,0)", showlakes=False,
                    landcolor="rgba(255,255,255,.06)", subunitcolor="rgba(4,18,27,.6)")
    fig = style_fig(fig, 360)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    return fig


def _geografia(df):
    st = _por_estado(df)
    top = st.head(4)
    perdida = st[st.Profit < 0].sort_values("Profit")
    desc = df.groupby("State").Discount.mean() * 100
    desc_perd, desc_resto = desc[perdida.index].mean(), desc.drop(perdida.index).mean()
    txt_ventas = (f"Las ventas se concentran en pocos estados: {', '.join(top.index)} suman el "
                  f"{top.share.sum():.0f} % del total, y solo California aporta el {top.share.iloc[0]:.0f} %. "
                  f"Son los estados más poblados y con las grandes ciudades de la costa oeste y el noreste. "
                  f"El modelo trabaja por región porque a nivel de estado muchas series tienen muy pocos pedidos.")
    txt_margen = (f"Vender más no siempre es ganar más. {len(perdida)} estados terminan con pérdida; los peores "
                  f"son {', '.join(f'{s} ({m:+.0f} %)' for s, m in perdida.margen.head(3).items())}, a pesar de que "
                  f"Texas es el tercer estado en ventas. La causa es el descuento: en los estados con pérdida "
                  f"el descuento promedio es {desc_perd:.0f} %, contra {desc_resto:.0f} % en el resto "
                  f"(ver la sección 6).")
    return html.Div([
        html.H4("5 · Geografía: ventas y margen por estado", className="section-title"),
        dbc.Row([
            dbc.Col(soft_card("Mapa: ventas totales por estado (2014–2017)", [
                _graph(_mapa(st, "Sales", SEQ_SCALE, "USD", "$,.0s")), insight(txt_ventas),
            ], "blue"), lg=6, className="mb-3"),
            dbc.Col(soft_card("Mapa: margen de ganancia por estado", [
                _graph(_mapa(st, "margen", DIV_SCALE, "Margen %", ".0f", zmid=0)), insight(txt_margen),
            ], "pink"), lg=6, className="mb-3"),
        ]),
    ])


# ------------------------------------------------------------ rentabilidad
def _correlacion(df):
    cols = {"Sales": "Ventas", "Quantity": "Cantidad", "Discount": "Descuento", "Profit": "Ganancia"}
    c = df[list(cols)].corr().rename(index=cols, columns=cols)
    fig = go.Figure(go.Heatmap(z=c.values, x=c.columns, y=c.index, colorscale=DIV_SCALE, zmin=-1, zmax=1,
                               text=c.round(2).values, texttemplate="%{text}", textfont=dict(size=14),
                               hovertemplate="%{y} vs. %{x}: %{z:.2f}<extra></extra>", colorbar=dict(thickness=12)))
    fig.update_yaxes(autorange="reversed")
    texto = (f"Correlación de Pearson entre las variables numéricas de cada línea de pedido. La ganancia sube "
             f"con las ventas ({c.loc['Ventas', 'Ganancia']:.2f}) y baja con el descuento "
             f"({c.loc['Descuento', 'Ganancia']:.2f}). El descuento casi no se relaciona con las ventas "
             f"({c.loc['Descuento', 'Ventas']:.2f}): dar más descuento no se traduce en pedidos más grandes.")
    return style_fig(fig, 340), texto


def _descuento_margen(df):
    d = df.assign(margen=df.Profit / df.Sales * 100)
    g = d.groupby("Discount").agg(margen=("margen", "mean"), pedidos=("Sales", "size"),
                                  perdida=("Profit", lambda x: (x < 0).mean() * 100)).reset_index()
    pend, inter = np.polyfit(d.Discount, d.margen, 1)
    xs = np.array([0, d.Discount.max()])
    fig = go.Figure([
        go.Scatter(x=g.Discount * 100, y=g.margen, mode="markers", name="Margen medio por nivel de descuento",
                   marker=dict(size=np.sqrt(g.pedidos) * 1.1 + 6, color=np.where(g.margen < 0, NEG, POS),
                               line=dict(width=1, color="rgba(4,18,27,.85)")),
                   customdata=np.stack([g.pedidos, g.perdida], axis=1),
                   hovertemplate="Descuento %{x:.0f} %<br>Margen medio %{y:.1f} %<br>Pedidos %{customdata[0]:,}"
                                 "<br>Con pérdida %{customdata[1]:.0f} %<extra></extra>"),
        go.Scatter(x=xs * 100, y=inter + pend * xs, mode="lines", name="Tendencia (mínimos cuadrados)",
                   line=dict(color=PROY, width=3)),
    ])
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,.4)")
    fig.update_xaxes(title="Descuento aplicado (%)", ticksuffix=" %")
    fig.update_yaxes(title="Margen de ganancia (%)", ticksuffix=" %")
    alto = d[d.Discount > 0.2]
    texto = (f"Cada burbuja es un nivel de descuento; su tamaño es el número de pedidos. Sin descuento el margen "
             f"medio es {g.margen.iloc[0]:.0f} % y ningún pedido pierde dinero; con descuentos mayores al 20 % "
             f"el {(alto.Profit < 0).mean() * 100:.0f} % de los pedidos ({len(alto):,}) termina con pérdida. La "
             f"línea de tendencia baja {abs(pend) / 10:.1f} puntos de margen por cada 10 puntos de descuento: "
             f"el descuento es el factor que más destruye la rentabilidad.")
    return style_fig(fig, 340), texto


def _burbujas_subcategoria(df):
    sc = df.groupby(["Category", "Sub-Category"]).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"),
                                                      pedidos=("Sales", "size")).reset_index()
    fig = go.Figure([go.Scatter(x=g.Sales, y=g.Profit, mode="markers+text", name=cat, text=g["Sub-Category"],
                                textposition="top center", textfont=dict(size=11),
                                marker=dict(size=np.sqrt(g.pedidos) * 1.3, color=CATEGORY_COLORS[cat], opacity=.8,
                                            line=dict(width=1, color="rgba(4,18,27,.85)")),
                                customdata=g.pedidos,
                                hovertemplate="<b>%{text}</b><br>Ventas $%{x:,.0f}<br>Ganancia $%{y:,.0f}"
                                              "<br>Pedidos %{customdata:,}<extra>" + cat + "</extra>")
                     for cat, g in sc.groupby("Category")])
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,.4)")
    fig.update_xaxes(title="Ventas totales (USD)", tickprefix="$", tickformat=",.0f")
    fig.update_yaxes(title="Ganancia total (USD)", tickprefix="$", tickformat=",.0f")
    neg = sc[sc.Profit < 0].sort_values("Profit")
    mejor = sc.sort_values("Profit", ascending=False).iloc[0]
    texto = (f"Gráfico de burbujas: cada una es una subcategoría, su tamaño es el número de pedidos. "
             f"Las que quedan bajo la línea punteada pierden dinero: {', '.join(neg['Sub-Category'])}. Tables es el "
             f"caso más claro: ${neg.Sales.iloc[0]:,.0f} en ventas y ${neg.Profit.iloc[0]:,.0f} de ganancia. "
             f"{mejor['Sub-Category']} es la más rentable (${mejor.Profit:,.0f}) con solo {mejor.pedidos} pedidos. "
             f"Para el negocio: revisar los descuentos de Tables y Bookcases antes de empujar más volumen.")
    return style_fig(fig, 420), texto


def _rentabilidad(df):
    fig_c, txt_c = _correlacion(df)
    fig_d, txt_d = _descuento_margen(df)
    fig_b, txt_b = _burbujas_subcategoria(df)
    return html.Div([
        html.H4("6 · Rentabilidad: descuento, ganancia y correlación", className="section-title"),
        insight("El modelo proyecta ventas; esta sección complementa el EDA con la ganancia (Profit), el descuento "
                "(Discount) y la cantidad (Quantity), que no entran al modelo pero explican qué ventas convienen."),
        dbc.Row([
            dbc.Col(soft_card("Mapa de calor: correlación entre variables numéricas", [
                _graph(fig_c), insight(txt_c),
            ], "lavender"), lg=5, className="mb-3 mt-3"),
            dbc.Col(soft_card("Dispersión con tendencia: descuento vs. margen", [
                _graph(fig_d), insight(txt_d),
            ], "pink"), lg=7, className="mb-3 mt-3"),
        ]),
        soft_card("Burbujas: ventas vs. ganancia por subcategoría", [
            _graph(fig_b), insight(txt_b),
        ], "green", height_full=False),
    ])


def layout():
    df = load_data()
    return html.Div([
        page_header("EDA", "Análisis univariado, bivariado, en el tiempo, de atípicos, geográfico y de rentabilidad: "
                    "qué factores explican el nivel de ventas"),
        html.Div([_univariado(df), _bivariado(df), _tiempo(df), _atipicos_modelo(df), _geografia(df),
                  _rentabilidad(df)], className="fade-in"),
    ])
