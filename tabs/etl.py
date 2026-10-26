"""tabs/etl.py — Extracción, transformación y carga: del CSV crudo a la tabla mensual del modelo."""
import dash_bootstrap_components as dbc
import plotly.express as px
from dash import dash_table, dcc, html

from common import (BRAND_LIGHT, CATEGORY_COLORS, TEXT, insight, kpi, load_data, load_mensual,
                    page_header, soft_card, style_fig, usd)

ETL_CODE = '''# --- Extracción
df = pd.read_csv("data/raw/Sample - Superstore.csv", encoding="latin1")

# --- Transformación
df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%m/%d/%Y")
df = df.rename(columns={"Order Date": "Order_Date"}).sort_values("Order_Date")

# Tabla mensual: ventas por categoría × región × mes (un mes sin pedidos queda en 0)
mensual = (df.groupby(["Category", "Region", pd.Grouper(key="Order_Date", freq="MS")])
             .Sales.sum().reindex(todas_las_combinaciones, fill_value=0))

# --- Carga
df.to_csv("data/superstore_transformado.csv", index=False)      # EDA
mensual.to_csv("data/ventas_mensuales.csv", index=False)       # modelo'''


def layout():
    df = load_data()
    m = load_mensual()
    ceros = int((m.Sales == 0).sum())

    serie = m.groupby(["Mes", "Category"], as_index=False).Sales.sum()
    fig = px.line(serie, x="Mes", y="Sales", color="Category", color_discrete_map=CATEGORY_COLORS,
                  labels={"Sales": "Ventas (USD)", "Mes": "", "Category": "Categoría"})
    fig.update_traces(line_width=2)

    muestra = m[m.Mes.dt.year == 2017].head(8).assign(Mes=lambda d: d.Mes.dt.strftime("%Y-%m"),
                                                       Sales=lambda d: d.Sales.map(lambda v: usd(v, 2)))
    tabla = dash_table.DataTable(
        muestra.to_dict("records"),
        [{"name": n, "id": c} for c, n in [("Category", "Categoría"), ("Region", "Región"), ("Mes", "Mes"),
                                           ("Sales", "Ventas (USD)")]],
        style_as_list_view=True, style_cell={"fontFamily": "Nunito, sans-serif", "padding": "6px", "color": TEXT},
        style_header={"backgroundColor": BRAND_LIGHT, "fontWeight": "700"},
    )

    return html.Div([
        page_header("ETL", "Extracción, transformación y carga: de las líneas de pedido a la tabla del modelo"),
        dbc.Row([
            dbc.Col(kpi("Líneas de pedido", f"{len(df):,}", "2014-01-03 a 2017-12-30"), md=3, className="mb-3"),
            dbc.Col(kpi("Valores faltantes", f"{int(df.isna().sum().sum())}", "en las 21 columnas"), md=3, className="mb-3"),
            dbc.Col(kpi("Filas duplicadas", f"{int(df.duplicated().sum())}", "registros repetidos"), md=3, className="mb-3"),
            dbc.Col(kpi("Tabla mensual", f"{len(m)} filas", f"12 series × 48 meses · {ceros} meses sin ventas"),
                    md=3, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(soft_card("Código del ETL (data/generate_data.py)", [
                dcc.Markdown(f"```python\n{ETL_CODE}\n```"),
                insight("Las fechas vienen como texto mes/día/año y se convierten a tipo fecha. La tabla "
                        "mensual incluye todas las combinaciones de categoría, región y mes: si en un mes "
                        "no hubo pedidos, la venta queda en 0 en vez de desaparecer, para que cada serie "
                        "tenga sus 48 meses completos."),
            ], "blue"), lg=7, className="mb-3"),
            dbc.Col(soft_card("Tabla mensual (muestra de 2017)", [
                tabla,
                insight("Cada fila es una categoría en una región durante un mes. Es la unidad que el "
                        "modelo proyecta: 3 categorías × 4 regiones = 12 series."),
            ], "lavender"), lg=5, className="mb-3"),
        ]),
        soft_card("Ventas mensuales por categoría (resultado de la tabla mensual)", [
            dcc.Graph(figure=style_fig(fig, 330), config={"displayModeBar": False}),
            insight("Las tres categorías repiten el mismo patrón cada año —ventas bajas a comienzos de año y "
                    "picos de septiembre a diciembre— y crecen desde 2016. Ese patrón repetible es lo que el "
                    "modelo aprende para proyectar."),
        ], "green", height_full=False),
    ])
