"""tabs/comparacion.py — Comparación de modelos, de métodos de búsqueda y validación de la proyección anual."""
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html

from common import (ACCENT, BRAND, BRAND_DARKER, PASTEL, REFERENCIA, aviso_sin_modelo, insight, kpi,
                    load_artifact, mejores_por_modelo, modelo_listo, page_header, soft_card, style_fig, usd)

CFG = {"displayModeBar": False}


def fig_ranking(res, ref, best):
    res = res.sort_values("MAE", ascending=False)
    colores = [BRAND_DARKER if m == best else ACCENT["blue"] for m in res.modelo]
    fig = go.Figure(go.Bar(x=res.MAE, y=res.modelo, orientation="h", marker_color=colores,
                           text=[usd(v) for v in res.MAE], textposition="inside", insidetextanchor="end",
                           textfont=dict(color="#fff")))
    fig.add_vline(x=ref.MAE, line_dash="dash", line_color=ACCENT["pink"],
                  annotation_text=f"Referencia {usd(ref.MAE)}", annotation_position="top")
    fig.update_xaxes(title="MAE en 2017 (USD, menor es mejor)", range=[0, ref.MAE * 1.18])
    return style_fig(fig, 320)


def fig_busqueda(r):
    fig = go.Figure()
    for metodo, color in [("Grid Search", BRAND), ("Random Search", PASTEL["peach"])]:
        s = r[r.busqueda == metodo]
        fig.add_trace(go.Bar(x=s.modelo, y=s.cv_MAE_indice, name=metodo, marker_color=color,
                             customdata=s[["evaluaciones", "tiempo_s"]],
                             hovertemplate="%{x}<br>Error CV %{y:.3f}<br>%{customdata[0]} combinaciones · "
                                           "%{customdata[1]:.1f} s<extra>" + metodo + "</extra>"))
    lo = r.cv_MAE_indice.min()
    fig.update_yaxes(title="Error en validación cruzada", range=[lo * 0.9, r.cv_MAE_indice.max() * 1.02])
    return style_fig(fig, 320)


def fig_residuos(best):
    p = load_artifact("predicciones.csv")
    t = p[p.conjunto == "Prueba"].groupby("Mes", as_index=False)[["Sales", best]].sum()
    t["residuo"] = t.Sales - t[best]
    fig = go.Figure(go.Bar(x=t.Mes, y=t.residuo, marker_color=[BRAND if v >= 0 else ACCENT["pink"] for v in t.residuo]))
    fig.add_hline(y=0, line_color="#9AAAB5")
    fig.update_yaxes(title="Real − proyectado (USD)", tickprefix="$", tickformat=",.0f")
    return style_fig(fig, 280)


def fig_hist_residuos(best):
    p = load_artifact("predicciones.csv")
    s = p[p.conjunto == "Prueba"]
    fig = go.Figure(go.Histogram(x=s.Sales - s[best], nbinsx=30, marker_color=BRAND, opacity=0.85))
    fig.add_vline(x=0, line_color=BRAND_DARKER, line_dash="dash")
    fig.update_xaxes(title="Real − proyectado por serie y mes (USD)", tickprefix="$")
    fig.update_yaxes(title="Frecuencia")
    return style_fig(fig, 280)


def fig_recursiva():
    rec = load_artifact("prueba_recursiva_2017.csv").groupby("Mes", as_index=False)[["Sales", "Pronostico"]].sum()
    pred = load_artifact("predicciones.csv")
    ref = pred[pred.conjunto == "Prueba"].groupby("Mes", as_index=False)[REFERENCIA].sum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec.Mes, y=rec.Sales, name="Real 2017", mode="lines+markers",
                             line=dict(color=BRAND_DARKER, width=2)))
    fig.add_trace(go.Scatter(x=rec.Mes, y=rec.Pronostico, name="Proyección a 12 meses (con datos hasta 2016)",
                             mode="lines", line=dict(color=ACCENT["pink"], width=2.5)))
    fig.add_trace(go.Scatter(x=ref.Mes, y=ref[REFERENCIA], name="Referencia (repetir 2016)", mode="lines",
                             line=dict(color="#9AAAB5", width=2, dash="dash")))
    fig.update_yaxes(title="Ventas del mes (USD)", tickprefix="$", tickformat=",.0f")
    return style_fig(fig, 340)


def layout():
    if not modelo_listo():
        return html.Div([page_header("Comparación"), aviso_sin_modelo()])
    resumen = load_artifact("resumen.json")
    r = load_artifact("resultados_modelos.csv")
    ref = r[r.modelo == REFERENCIA].iloc[0]
    r = r[r.modelo != REFERENCIA]
    res = mejores_por_modelo()
    best = resumen["mejor_modelo"]
    rec, rf = resumen["prueba_recursiva"], resumen["referencia"]

    g = r.groupby("busqueda").agg(tiempo=("tiempo_s", "sum"), evals=("evaluaciones", "sum"))
    pares = r.pivot(index="modelo", columns="busqueda", values="cv_MAE_indice")
    gana_random = pares.index[pares["Random Search"] < pares["Grid Search"]].tolist()

    return html.Div([
        page_header("Comparación", "Qué modelo proyecta mejor, qué método de búsqueda conviene y cuánto "
                    "confiar en la proyección anual"),
        dbc.Row([
            dbc.Col(soft_card("Ranking de modelos (MAE en 2017)", [
                dcc.Graph(figure=fig_ranking(res, ref, best), config=CFG),
                insight(f"Todos los modelos con MAE a la izquierda de la línea superan a la referencia. Las "
                        f"diferencias entre ellos son pequeñas (de {usd(res.MAE.min())} a {usd(res.MAE.max())}): "
                        f"con 288 filas de entrenamiento, la señal disponible es sobre todo la estacionalidad, "
                        f"y todos la capturan. {best} (oscuro) se eligió por su error de validación cruzada, "
                        f"antes de mirar 2017."),
            ], "blue"), lg=6, className="mb-3"),
            dbc.Col(soft_card("Grid Search vs. Random Search", [
                dcc.Graph(figure=fig_busqueda(r), config=CFG),
                insight(f"Grid Search evaluó {int(g.loc['Grid Search', 'evals'])} combinaciones en "
                        f"{g.loc['Grid Search', 'tiempo']:.1f} s; Random Search, "
                        f"{int(g.loc['Random Search', 'evals'])} (12 por modelo) en "
                        f"{g.loc['Random Search', 'tiempo']:.1f} s. Random Search logró menor error de validación "
                        f"en {len(gana_random)} de 7 modelos ({', '.join(gana_random)}): al muestrear rangos "
                        f"continuos prueba valores que una rejilla fija no incluye. Con un presupuesto similar, "
                        f"ninguno de los dos domina, y las diferencias son pequeñas frente al error del modelo."),
            ], "lavender"), lg=6, className="mb-3"),
        ]),
        html.H4("¿Cuánto confiar en la proyección de un año completo?", className="section-title"),
        dbc.Row([
            dbc.Col(kpi("Desvío del total anual", f"{rec['total_desvio_total_pct']:+.1f} %",
                        f"referencia: {rf['total_desvio_total_pct']:+.1f} %"), md=4, className="mb-3"),
            dbc.Col(kpi("Error del total mensual", f"{rec['total_MAE_pct']:.1f} %",
                        f"{usd(rec['total_MAE'])} por mes en promedio"), md=4, className="mb-3"),
            dbc.Col(kpi("R² del total mensual", f"{rec['total_R2']:.2f}", "variación mensual explicada"),
                    md=4, className="mb-3"),
        ]),
        soft_card("Prueba de la proyección a 12 meses: 2017 proyectado solo con datos hasta 2016", [
            dcc.Graph(figure=fig_recursiva(), config=CFG),
            insight("Es la misma forma en que se proyecta 2018: cada mes proyectado se usa como historia "
                    "del siguiente, sin conocer ningún dato real del año. El modelo acierta el total del año "
                    "y la forma estacional; los errores están en el detalle de cada mes. La referencia "
                    "subestima todo el año porque repite 2016 sin el crecimiento."),
        ], "green", height_full=False),
        html.H4(f"Análisis de residuos ({best}, prueba 2017)", className="section-title"),
        dbc.Row([
            dbc.Col(soft_card("Residuos del total mensual", [
                dcc.Graph(figure=fig_residuos(best), config=CFG),
                insight(f"Barras arriba de cero: el mes vendió más de lo proyectado. Hay meses por encima y "
                        f"por debajo, y en el año se compensan (el total quedó a "
                        f"{resumen['prueba']['total_desvio_total_pct']:+.1f} % del real). Los errores más grandes "
                        f"están en noviembre y diciembre, los meses de mayor venta, donde los pedidos grandes de "
                        f"fin de año cambian de un mes a otro."),
            ], "peach"), lg=6, className="mb-3"),
            dbc.Col(soft_card("Distribución de residuos por serie", [
                dcc.Graph(figure=fig_hist_residuos(best), config=CFG),
                insight("La mayoría de los errores se concentran cerca de cero, con una cola de meses que "
                        "vendieron mucho más de lo esperado: pedidos grandes puntuales que ningún modelo "
                        "puede anticipar con estas variables."),
            ], "peach"), lg=6, className="mb-3"),
        ]),
    ])
