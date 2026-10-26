"""tabs/comparacion.py — Ranking de modelos frente a la referencia y validación de la proyección anual."""
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html

from common import (POS, PROY, REAL, REF, REFERENCIA, aviso_sin_modelo, insight, kpi, load_artifact,
                    mejores_por_modelo, modelo_listo, page_header, soft_card, style_fig, usd)

CFG = {"displayModeBar": False}


def fig_ranking(res, ref, best):
    res = res.sort_values("MAE", ascending=False)
    colores = [POS if m == best else REAL for m in res.modelo]
    fig = go.Figure(go.Bar(x=res.MAE, y=res.modelo, orientation="h", marker_color=colores,
                           text=[usd(v) for v in res.MAE], textposition="inside", insidetextanchor="end",
                           textfont=dict(color="#04121b")))
    fig.add_vline(x=ref.MAE, line_dash="dash", line_color=REF,
                  annotation_text=f"Referencia {usd(ref.MAE)}", annotation_position="top")
    fig.update_xaxes(title="MAE en 2017 (USD, menor es mejor)", range=[0, ref.MAE * 1.18])
    return style_fig(fig, 320)


def fig_recursiva():
    rec = load_artifact("prueba_recursiva_2017.csv").groupby("Mes", as_index=False)[["Sales", "Proyeccion"]].sum()
    pred = load_artifact("predicciones.csv")
    ref = pred[pred.conjunto == "Prueba"].groupby("Mes", as_index=False)[REFERENCIA].sum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec.Mes, y=rec.Sales, name="Real 2017", mode="lines+markers",
                             line=dict(color=REAL, width=2)))
    fig.add_trace(go.Scatter(x=rec.Mes, y=rec.Proyeccion, name="Proyección a 12 meses (con datos hasta 2016)",
                             mode="lines", line=dict(color=PROY, width=2.5)))
    fig.add_trace(go.Scatter(x=ref.Mes, y=ref[REFERENCIA], name="Referencia (repetir 2016)", mode="lines",
                             line=dict(color=REF, width=2, dash="dash")))
    fig.update_yaxes(title="Ventas del mes (USD)", tickprefix="$", tickformat=",.0f")
    return style_fig(fig, 340)


def layout():
    if not modelo_listo():
        return html.Div([page_header("Comparación"), aviso_sin_modelo()])
    resumen = load_artifact("resumen.json")
    r = load_artifact("resultados_modelos.csv")
    ref = r[r.modelo == REFERENCIA].iloc[0]
    res = mejores_por_modelo()
    best = resumen["mejor_modelo"]
    rec, rf = resumen["prueba_recursiva"], resumen["referencia"]

    return html.Div([
        page_header("Comparación", "Qué modelo proyecta mejor y cuánto confiar en la proyección anual"),
        soft_card("Ranking de modelos (MAE en 2017)", [
            dcc.Graph(figure=fig_ranking(res, ref, best), config=CFG),
            insight(f"Todos los modelos con MAE a la izquierda de la línea superan a la referencia. Las "
                    f"diferencias entre ellos son pequeñas (de {usd(res.MAE.min())} a {usd(res.MAE.max())}): "
                    f"con 288 filas de entrenamiento, la señal disponible es sobre todo la estacionalidad, "
                    f"y todos la capturan. {best} (en verde) se eligió por su error de validación cruzada, "
                    f"antes de mirar 2017."),
        ], "blue", height_full=False),
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
    ])
