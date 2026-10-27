"""tabs/modelos.py — Modelos implementados y resultados en el periodo de prueba (2017)."""
import json

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from common import (BRAND, BRAND_LIGHT, PROY, REAL, REF, REFERENCIA, aviso_sin_modelo, insight, kpi,
                    load_artifact, mejores_por_modelo, modelo_listo, page_header, soft_card, style_fig, usd)

CFG = {"displayModeBar": False}


def _tabla(res, ref):
    filas = []
    for _, f in res.iterrows():
        params = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in json.loads(f.hiperparametros).items())
        filas.append(html.Tr([html.Td(html.B(f.modelo)), html.Td(usd(f.MAE)),
                              html.Td(usd(f.RMSE)), html.Td(f"{f.R2:.2f}"), html.Td(f"{f.total_MAE_pct:.1f} %"),
                              html.Td(html.Small(params, className="text-muted"))]))
    filas.append(html.Tr([html.Td(html.I(REFERENCIA)), html.Td(usd(ref.MAE)), html.Td(usd(ref.RMSE)),
                          html.Td(f"{ref.R2:.2f}"), html.Td(f"{ref.total_MAE_pct:.1f} %"), html.Td("—")]))
    enc = ["Modelo", "MAE", "RMSE", "R²", "Error total mensual", "Hiperparámetros"]
    return dbc.Table([html.Thead(html.Tr([html.Th(h) for h in enc])), html.Tbody(filas)],
                     hover=True, responsive=True, borderless=True, className="mb-0")


def fig_ajuste(modelo):
    """Total mensual de la empresa: real vs. proyectado, en entrenamiento y prueba."""
    p = load_artifact("predicciones.csv")
    t = p.groupby(["Mes", "conjunto"], as_index=False)[["Sales", modelo]].sum()
    fig = go.Figure()
    ini, fin = t[t.conjunto == "Prueba"].Mes.min(), t.Mes.max()
    fig.add_vrect(x0=ini, x1=fin, fillcolor=BRAND_LIGHT, opacity=0.7, line_width=0, layer="below",
                  annotation_text="Prueba (2017)", annotation_position="top left")
    fig.add_trace(go.Scatter(x=t.Mes, y=t.Sales, name="Real", mode="lines+markers",
                             line=dict(color=REAL, width=2), marker_size=5))
    for conj, dash in [("Entrenamiento", "dot"), ("Prueba", "solid")]:
        s = t[t.conjunto == conj]
        fig.add_trace(go.Scatter(x=s.Mes, y=s[modelo], name=f"Proyección ({conj.lower()})", mode="lines",
                                 line=dict(color=PROY, width=2.5, dash=dash)))
    fig.update_yaxes(title="Ventas del mes (USD)", tickprefix="$", tickformat=",.0f")
    return style_fig(fig, 360)


def _fmt(v):
    return "None" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))


def _tabla_rejillas(res):
    """Rejilla de hiperparámetros que probó GridSearchCV en cada modelo y el valor elegido."""
    filas = []
    for _, f in res.iterrows():
        rejilla = json.loads(f.rejilla)
        mejor = json.loads(f.hiperparametros)
        filas.append(html.Tr([
            html.Td(html.B(f.modelo)), html.Td(int(f.evaluaciones)),
            html.Td(html.Small("; ".join(f"{k} ∈ {{{', '.join(_fmt(x) for x in v)}}}" for k, v in rejilla.items()),
                               className="text-muted")),
            html.Td(html.Small(", ".join(f"{k}={_fmt(v)}" for k, v in mejor.items()))),
        ]))
    enc = ["Modelo", "Combinaciones", "Rejilla probada", "Elegido por validación cruzada"]
    return dbc.Table([html.Thead(html.Tr([html.Th(h) for h in enc])), html.Tbody(filas)],
                     hover=True, responsive=True, borderless=True, className="mb-0")


def errores_por_conjunto():
    """MAE de cada modelo en entrenamiento y en prueba, como % de la venta media de ese periodo.

    En porcentaje porque 2017 vende más que 2015–2016: en dólares el error de prueba sube aunque el
    modelo no haya empeorado (le pasa incluso a la referencia)."""
    p = load_artifact("predicciones.csv")
    nombres = [c for c in p.columns if c not in ("Category", "Region", "Mes", "conjunto", "Sales")]
    filas = []
    for m in nombres:
        fila = {"modelo": m}
        for conj in ("Entrenamiento", "Prueba"):
            s = p[p.conjunto == conj]
            fila[conj] = (s.Sales - s[m]).abs().mean() / s.Sales.mean() * 100
        filas.append(fila)
    t = pd.DataFrame(filas)
    t["brecha"] = t.Prueba - t.Entrenamiento
    return t.sort_values("Prueba")


def fig_sobreajuste(t):
    fig = go.Figure([
        go.Bar(x=t.modelo, y=t.Entrenamiento, name="Entrenamiento (2015–2016)", marker_color=BRAND_LIGHT,
               marker_line=dict(color=REAL, width=1.5), hovertemplate="%{x}<br>%{y:.1f} %<extra>Entrenamiento</extra>"),
        go.Bar(x=t.modelo, y=t.Prueba, name="Prueba (2017)", marker_color=REAL,
               hovertemplate="%{x}<br>%{y:.1f} %<extra>Prueba</extra>"),
    ])
    fig.update_yaxes(title="MAE (% de la venta media por serie)", ticksuffix=" %")
    fig.update_xaxes(tickangle=-20)
    return style_fig(fig, 340)


def fig_importancia():
    imp = load_artifact("importancia.csv").sort_values("importancia")
    fig = go.Figure(go.Bar(x=imp.importancia, y=imp.nombre, orientation="h", marker_color=BRAND,
                           error_x=dict(type="data", array=imp["std"], color=REF, thickness=1)))
    fig.update_xaxes(title="Aumento del error al desordenar la variable")
    return style_fig(fig, 300)


def layout():
    if not modelo_listo():
        return html.Div([page_header("Modelos y resultados"), aviso_sin_modelo()])
    resumen = load_artifact("resumen.json")
    res = mejores_por_modelo()
    r = load_artifact("resultados_modelos.csv")
    ref = r[r.modelo == REFERENCIA].iloc[0]
    best = resumen["mejor_modelo"]
    p, rf = resumen["prueba"], resumen["referencia"]
    mejora = (1 - p["MAE"] / rf["MAE"]) * 100
    opciones = [{"label": m, "value": m} for m in res.modelo] + [{"label": REFERENCIA, "value": REFERENCIA}]
    sob = errores_por_conjunto()
    sm = sob[sob.modelo != REFERENCIA]
    b, rr = sob.set_index("modelo").loc[best], sob.set_index("modelo").loc[REFERENCIA]
    mayor = sm.loc[sm.brecha.idxmax()]
    estables = sm[sm.brecha <= 1].modelo.tolist()
    imp = load_artifact("importancia.csv")
    sin_aporte = imp[imp.importancia <= 0].nombre.tolist()

    return html.Div([
        page_header("Modelos y resultados", "7 modelos de regresión evaluados en 2017, un año que no vieron "
                    "al entrenar"),
        dbc.Row([
            dbc.Col(kpi("Mejor modelo", best, "elegido por validación cruzada (Grid Search)"),
                    md=3, className="mb-3"),
            dbc.Col(kpi("MAE en 2017", usd(p["MAE"]), f"{mejora:.0f} % menos error que la referencia "
                        f"({usd(rf['MAE'])})"), md=3, className="mb-3"),
            dbc.Col(kpi("Error en el total mensual", f"{p['total_MAE_pct']:.1f} %",
                        f"referencia: {rf['total_MAE_pct']:.1f} % · R² {p['total_R2']:.2f}"), md=3, className="mb-3"),
            dbc.Col(kpi("Desvío del total 2017", f"{p['total_desvio_total_pct']:+.1f} %",
                        f"referencia: {rf['total_desvio_total_pct']:+.1f} %"), md=3, className="mb-3"),
        ]),
        soft_card("Real vs. proyectado: ventas totales de la empresa por mes", [
            dbc.Row(dbc.Col(dcc.Dropdown(opciones, best, id="modelos-sel", clearable=False), md=4),
                    className="mb-2"),
            dcc.Graph(id="modelos-ajuste", config=CFG),
            insight("Cada punto es la suma de las 12 series del mes. La zona sombreada es 2017: el modelo no "
                    "vio esos meses al entrenar, cada mes se proyecta con la historia hasta el mes anterior. "
                    "Elija otro modelo para comparar: la referencia (repetir el año anterior) queda por debajo "
                    "porque no capta el crecimiento de 2017."),
        ], "blue", height_full=False),
        html.H4("Resultados por modelo (prueba 2017)", className="section-title"),
        soft_card("Métricas por categoría × región × mes", [
            _tabla(res, ref),
            insight(f"MAE y RMSE miden el error de cada serie mensual (la venta media por serie y mes en 2017 "
                    f"fue de {usd(resumen['ventas_2017'] / 144)}), por eso su R² es bajo: una categoría en una "
                    f"región varía mucho de un "
                    f"mes a otro. Al sumar las series, el error del total mensual de la empresa baja a "
                    f"{p['total_MAE_pct']:.0f} %. {(res.MAE < ref.MAE).sum()} de {len(res)} modelos superan "
                    f"a la referencia en MAE."),
        ], "lavender", height_full=False),
        html.H4("Hiperparámetros probados con Grid Search", className="section-title"),
        soft_card("Rejilla de cada modelo y valor elegido", [
            _tabla_rejillas(res),
            insight("GridSearchCV entrenó cada combinación de la rejilla con TimeSeriesSplit (4 pliegues) y se "
                    "quedó con la de menor MAE en validación cruzada. Los hiperparámetros controlan la "
                    "complejidad: alpha en Ridge y Lasso, vecinos en KNN, profundidad y tamaño de hoja en los "
                    "árboles, tasa de aprendizaje en XGBoost y C y ε en SVR."),
        ], "lavender", height_full=False),
        html.H4("¿Hay sobreajuste? Error en entrenamiento vs. prueba", className="section-title"),
        soft_card("MAE en entrenamiento y en prueba", [
            dcc.Graph(figure=fig_sobreajuste(sob), config=CFG),
            insight(f"Un modelo sobreajustado se equivoca poco con los datos que vio y mucho con los nuevos. "
                    f"Sin brecha relevante (1 punto o menos): {', '.join(estables)} y la "
                    f"referencia. Con brecha: los modelos más flexibles, que memorizan parte del entrenamiento; "
                    f"la mayor es la de {mayor.modelo} ({mayor.Entrenamiento:.0f} % → {mayor.Prueba:.0f} %). "
                    f"{best} pasa de {b.Entrenamiento:.0f} % a {b.Prueba:.0f} % (+{b.brecha:.0f} puntos): un "
                    f"sobreajuste moderado, pero en 2017 sigue entre los de menor error y por debajo de la "
                    f"referencia ({rr.Prueba:.0f} %). La validación cruzada mide el error en meses no vistos, así "
                    f"que al elegir el modelo ya tiene en cuenta este efecto."),
        ], "peach", height_full=False),
        html.H4(f"Importancia de variables ({best})", className="section-title"),
        soft_card("Qué usa el modelo para proyectar", [
            dcc.Graph(figure=fig_importancia(), config=CFG),
            insight("Importancia por permutación en 2017: cuánto aumenta el error si se desordena cada "
                    "variable. Dominan el mes del año y las ventas del mismo mes del año anterior, es decir, "
                    "la estacionalidad que mostró el EDA. "
                    + (f"{', '.join(sin_aporte)} tienen importancia cero o negativa: desordenarlas no empeora "
                       f"el modelo (incluso lo mejora un poco por azar), o sea que no aportan información útil en "
                       f"2017. Categoría y región pesan poco porque el índice ya pone cada serie en su propia "
                       f"escala." if sin_aporte else
                       "Categoría y región pesan poco porque el índice ya pone cada serie en su propia escala.")),
        ], "peach", height_full=False),
    ])


@callback(Output("modelos-ajuste", "figure"), Input("modelos-sel", "value"))
def actualizar_ajuste(modelo):
    return fig_ajuste(modelo)
