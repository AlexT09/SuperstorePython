"""tabs/recomendaciones.py — Recomendaciones aplicadas: cómo cambiaría la proyección 2018 si el negocio
aplica las recomendaciones del EDA.

Proyección con recomendaciones = proyección base 2018 (Random Forest, model/train_model.py)
                                  × efecto de cada recomendación activada.

- Tope de descuento: efecto estimado con el modelo de margen por línea (model/train_descuentos.py),
  simulado sobre los pedidos de 2017 y llevado a 2018 como un factor por categoría × región.
- Temporada alta e impulso de una categoría: supuestos que fija el usuario.
La ganancia base de 2018 usa el margen de 2017 de cada categoría × región.
"""
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from common import (ART, NEG, POS, PROY, REAL, REF, insight, kpi, load_artifact, load_mensual, modelo_listo,
                    page_header, soft_card, style_fig, usd)

CFG = {"displayModeBar": False}
TODAS = "Todas"
CATS = ["Furniture", "Office Supplies", "Technology"]
REGIONS = ["Central", "East", "South", "West"]
TOPES = [0, 10, 15, 20, 30, 40, 50, 60, 70, 80]  # % de descuento que existen en los datos
CONTROLES = [Input(i, "value") for i in ["rec-tope-on", "rec-tope", "rec-tope-reg", "rec-perdida", "rec-temp-on",
                                         "rec-temp", "rec-cat-on", "rec-cat", "rec-cat-pct"]]


def listo() -> bool:
    return modelo_listo() and (ART / "descuentos_escenarios.csv").exists()


def _escenarios():
    e = load_artifact("descuentos_escenarios.csv").copy()
    e["tope"] = (e.tope * 100).round().astype(int)
    return e


def _efecto_tope(tope, region, perdida):
    """Por categoría × región: factor de ventas y margen después del tope (sobre los pedidos de 2017).

    perdida = fracción de las líneas con descuento recortado que ya no se venden (supuesto)."""
    e = _escenarios()
    e = e[e.tope == tope].set_index(["Category", "Region"])
    aplica = e.index.get_level_values("Region") == region if region != TODAS else np.ones(len(e), bool)
    ventas = np.where(aplica, e.ventas_base - e.ventas_cambian_antes + e.ventas_cambian_despues * (1 - perdida),
                      e.ventas_base)
    ganancia = np.where(aplica, e.ganancia_base - e.ganancia_cambian_antes + e.ganancia_cambian_despues * (1 - perdida),
                        e.ganancia_base)
    return pd.DataFrame({"margen_base": e.ganancia_base / e.ventas_base, "factor_ventas": ventas / e.ventas_base,
                         "margen_sim": ganancia / ventas, "aplica": aplica}, index=e.index)


def simular(tope_on, tope, region, perdida, temp_on, temp, cat_on, cat, cat_pct):
    """Proyección 2018 por categoría × región × mes, base y con cada recomendación aplicada en orden."""
    p = load_artifact("proyeccion_2018.csv").copy()
    ef = _efecto_tope(tope if tope_on else 80, region, perdida / 100)
    p = p.join(ef, on=["Category", "Region"])
    p["v_base"] = p.Proyeccion
    p["g_base"] = p.v_base * p.margen_base
    # 1) Tope de descuento: cambia ventas (precio y pedidos perdidos) y margen
    p["v1"] = p.v_base * p.factor_ventas
    p["g1"] = p.v1 * p.margen_sim
    # 2) Temporada alta: más ventas de septiembre a diciembre, con el mismo margen
    alta = p.Mes.dt.month >= 9
    p["v2"] = p.v1 * np.where(temp_on & alta, 1 + temp / 100, 1)
    p["g2"] = p.v2 * p.margen_sim
    # 3) Impulso de una categoría
    p["v3"] = p.v2 * np.where(cat_on & (p.Category == cat), 1 + cat_pct / 100, 1)
    p["g3"] = p.v3 * p.margen_sim
    return p


def equilibrio(tope, region):
    """Desde qué % de pedidos perdidos el tope deja de mejorar la ganancia y las ventas (líneas afectadas)."""
    e = _escenarios()
    e = e[e.tope == tope]
    if region != TODAS:
        e = e[e.Region == region]
    out = {}
    for k in ["ganancia", "ventas"]:
        antes, despues = e[f"{k}_cambian_antes"].sum(), e[f"{k}_cambian_despues"].sum()
        g0, g1 = despues - antes, -antes  # cambio si no se pierde nada / si se pierden todos esos pedidos
        out[k] = None if g0 <= 0 else (1.0 if g1 >= 0 else g0 / (g0 - g1))
    out["lineas"] = int(e.lineas_cambian.sum())
    return out


# ------------------------------------------------------------------ gráficos
def fig_mensual(p):
    real = load_mensual()
    real = real[real.Mes.dt.year >= 2015].groupby("Mes", as_index=False).Sales.sum()
    m = p.groupby("Mes", as_index=False)[["v_base", "v3"]].sum()
    ult = real.tail(1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=real.Mes, y=real.Sales, name="Real", mode="lines+markers",
                             line=dict(color=REAL, width=2), marker_size=4))
    fig.add_trace(go.Scatter(x=list(ult.Mes) + list(m.Mes), y=list(ult.Sales) + list(m.v_base),
                             name="Proyección base 2018", mode="lines+markers",
                             line=dict(color=PROY, width=2.5, dash="dash"), marker_size=5))
    fig.add_trace(go.Scatter(x=list(ult.Mes) + list(m.Mes), y=list(ult.Sales) + list(m.v3),
                             name="Con recomendaciones", mode="lines+markers",
                             line=dict(color=POS, width=3), marker_size=6))
    fig.update_yaxes(title="Ventas del mes (USD)", tickprefix="$", tickformat=",.0f")
    return style_fig(fig, 360)


def fig_aportes(p):
    """Cascada: de la ganancia base 2018 a la ganancia con recomendaciones, paso a paso."""
    g = p[["g_base", "g1", "g2", "g3"]].sum()
    pasos = ["Ganancia base", "Tope de descuento", "Temporada alta", "Impulso de categoría", "Con recomendaciones"]
    valores = [g.g_base, g.g1 - g.g_base, g.g2 - g.g1, g.g3 - g.g2, g.g3]
    fig = go.Figure(go.Waterfall(
        x=pasos, y=valores, measure=["absolute", "relative", "relative", "relative", "total"],
        text=[usd(valores[0])] + [f"{v:+,.0f}".replace("+", "+$").replace("-", "−$") for v in valores[1:4]]
        + [usd(valores[4])], textposition="outside",
        increasing=dict(marker_color=POS), decreasing=dict(marker_color=NEG), totals=dict(marker_color=PROY),
        connector=dict(line=dict(color=REF, width=1, dash="dot"))))
    fig.update_yaxes(title="Ganancia 2018 (USD)", tickprefix="$", tickformat=",.0f",
                     range=[0, max(np.cumsum(valores[:4]).max(), valores[4]) * 1.18])
    fig = style_fig(fig, 340)
    fig.update_layout(showlegend=False)
    return fig


def fig_curva(region, perdida, tope_actual):
    """Ganancia de 2017 con cada tope de descuento: dónde conviene ponerlo."""
    filas = []
    for t in TOPES:
        ef = _efecto_tope(t, region, perdida / 100)
        e = _escenarios()
        e = e[e.tope == t].set_index(["Category", "Region"])
        ventas = e.ventas_base * ef.factor_ventas
        filas.append({"tope": t, "ganancia": (ventas * ef.margen_sim).sum(), "ventas": ventas.sum()})
    c = pd.DataFrame(filas)
    base = c[c.tope == 80].ganancia.iloc[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c.tope, y=c.ganancia, mode="lines+markers", name="Ganancia 2017 simulada",
                             line=dict(color=POS, width=3), marker_size=7,
                             hovertemplate="Tope %{x} %<br>Ganancia %{y:$,.0f}<extra></extra>"))
    fig.add_hline(y=base, line_dash="dash", line_color=REF, annotation_text=f"Real 2017: {usd(base)}",
                  annotation_position="bottom right")
    if tope_actual is not None:
        fig.add_vline(x=tope_actual, line_dash="dot", line_color=PROY,
                      annotation_text=f"Tope elegido: {tope_actual} %", annotation_position="top")
    fig.update_xaxes(title="Tope de descuento (%)", ticksuffix=" %", tickvals=TOPES)
    fig.update_yaxes(title="Ganancia (USD)", tickprefix="$", tickformat=",.0f")
    return style_fig(fig, 320), c


# ------------------------------------------------------------------ layout
def _modelo_descuentos():
    r = load_artifact("descuentos_resultados.csv")
    s = load_artifact("descuentos_resumen.json")
    ref = r.iloc[0]
    filas = [html.Tr([html.Td(html.B(f.modelo) if f.modelo == s["mejor_modelo"] else f.modelo),
                      html.Td("—" if pd.isna(f.cv_MAE_margen_pts) else f"{f.cv_MAE_margen_pts:.2f}"),
                      html.Td(f"{f.MAE_margen_pts:.2f}"), html.Td(f"{f.desvio_ganancia_pct:+.1f} %")])
             for f in r.iloc[1:].sort_values("cv_MAE_margen_pts").itertuples()]
    filas.append(html.Tr([html.Td(html.I(ref.modelo)), html.Td("—"), html.Td(f"{ref.MAE_margen_pts:.2f}"),
                          html.Td(f"{ref.desvio_ganancia_pct:+.1f} %")]))
    enc = ["Modelo", "Error CV (puntos de margen)", "Error en 2017 (puntos)", "Desvío de la ganancia 2017"]
    tabla = dbc.Table([html.Thead(html.Tr([html.Th(h) for h in enc])), html.Tbody(filas)],
                      hover=True, responsive=True, borderless=True, className="mb-0")
    p = s["prueba"]
    return soft_card("Cómo se estima el efecto del descuento", [
        tabla,
        insight(f"Cada fila de entrenamiento es una línea de pedido ({s['lineas_entrenamiento']:,} de 2014–2016) y el "
                f"modelo predice su margen (ganancia ÷ venta) según descuento, cantidad, subcategoría, región y "
                f"segmento. Se eligió {s['mejor_modelo']} por validación cruzada, igual que en la proyección. En "
                f"2017 se equivoca en {p['MAE_margen_pts']:.1f} puntos de margen por línea, casi lo mismo que la "
                f"referencia ({ref.MAE_margen_pts:.1f}): no la supera en precisión. Se usa porque la referencia "
                f"solo conoce combinaciones que ya ocurrieron y el simulador necesita otras (por ejemplo, Binders "
                f"con 20 % de descuento en una región donde siempre se vendió con 70 %). Ninguno de los dos prueba "
                f"causa: estiman lo que suele pasar en los datos."),
    ], "lavender", height_full=False)


def _control(titulo, switch_id, valor, cuerpo):
    return html.Div([dbc.Switch(id=switch_id, label=html.B(titulo), value=valor, className="mb-2"), cuerpo],
                    className="mb-3")


def _slider(id_, minimo, maximo, valor, paso=None, marcas=None):
    marcas = marcas or {v: f"{v} %" for v in range(minimo, maximo + 1, max((maximo - minimo) // 4, 1))}
    return dcc.Slider(id=id_, min=minimo, max=maximo, value=valor, step=paso, marks=marcas,
                      tooltip={"placement": "bottom", "template": "{value} %"})


def layout():
    if not listo():
        return html.Div([page_header("Recomendaciones aplicadas"), soft_card("Modelos no entrenados", html.P([
            "Ejecute ", html.Code("python model/train_model.py"), " y ", html.Code("python model/train_descuentos.py"),
            " y vuelva a cargar la página."], className="mb-0"), height_full=False)])
    controles = soft_card("Recomendaciones a aplicar", [
        _control("Tope de descuento", "rec-tope-on", True, html.Div([
            html.Label("Descuento máximo", className="small"),
            _slider("rec-tope", 0, 80, 20, paso=None, marcas={t: f"{t}" for t in TOPES}),
            html.Label("Dónde se aplica", className="small mt-2"),
            dcc.Dropdown([TODAS] + REGIONS, TODAS, id="rec-tope-reg", clearable=False),
            html.Label("Supuesto: pedidos que se pierden al recortar el descuento", className="small mt-2"),
            _slider("rec-perdida", 0, 100, 25, paso=5),
        ])),
        html.Hr(className="my-2"),
        _control("Reforzar la temporada alta (septiembre a diciembre)", "rec-temp-on", False, html.Div([
            html.Label("Supuesto: aumento de ventas en esos meses", className="small"),
            _slider("rec-temp", 0, 20, 5, paso=1),
        ])),
        html.Hr(className="my-2"),
        _control("Impulsar una categoría con campañas", "rec-cat-on", False, html.Div([
            dcc.Dropdown(CATS, "Technology", id="rec-cat", clearable=False, className="mb-2"),
            html.Label("Supuesto: aumento de ventas de esa categoría", className="small"),
            _slider("rec-cat-pct", 0, 20, 5, paso=1),
        ])),
        insight("El efecto del tope sale del modelo de descuentos. Los aumentos por temporada y por categoría "
                "son supuestos: el modelo no puede estimarlos porque los datos no registran campañas."),
    ], "peach")

    return html.Div([
        page_header("Recomendaciones aplicadas", "Cómo cambiaría la proyección 2018 si el negocio aplica las "
                    "recomendaciones del EDA"),
        dbc.Row([
            dbc.Col(controles, lg=4, className="mb-3"),
            dbc.Col([
                dbc.Row(id="rec-kpis"),
                html.Div(soft_card("Impacto de las recomendaciones en el negocio", html.Div(id="rec-lectura"),
                                   "green", height_full=False), className="mb-3"),
                soft_card("Ventas mensuales: real, proyección base y con recomendaciones", [
                    dcc.Graph(id="rec-fig", config=CFG),
                ], "blue", height_full=False),
            ], lg=8, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(soft_card("Aporte de cada recomendación a la ganancia 2018", [
                dcc.Graph(id="rec-aportes", config=CFG),
                insight("Cada barra suma o resta a la ganancia base. Se aplican en orden: el tope cambia el margen "
                        "y las demás recomendaciones venden más con ese margen nuevo."),
            ], "green"), lg=6, className="mb-3"),
            dbc.Col(soft_card("¿Dónde conviene poner el tope?", [
                dcc.Graph(id="rec-curva", config=CFG), html.Div(id="rec-curva-lectura"),
            ], "yellow"), lg=6, className="mb-3"),
        ]),
        _modelo_descuentos(),
        soft_card("Supuestos y límites", html.Ul([
            html.Li("La proyección base es la de Random Forest (pestaña Proyección). La ganancia base de 2018 usa el "
                    "margen de 2017 de cada categoría en cada región."),
            html.Li("Con menos descuento, cada pedido conserva sus unidades y paga más por ellas: en los datos el "
                    "descuento no se relaciona con la cantidad por línea (correlación 0,01)."),
            html.Li("Lo que los datos no dicen es cuántos clientes dejarían de comprar sin el descuento. Por eso es un "
                    "supuesto ajustable, y la lectura indica hasta qué pérdida conviene el tope."),
            html.Li("Son escenarios estimados, no causas probadas. Antes de aplicar una recomendación en toda la "
                    "empresa, conviene probarla en una región y comparar con las demás."),
        ], className="mb-0"), "green", height_full=False),
    ])


# ------------------------------------------------------------------ callbacks
@callback(Output("rec-tope", "disabled"), Output("rec-tope-reg", "disabled"), Output("rec-perdida", "disabled"),
          Output("rec-temp", "disabled"), Output("rec-cat", "disabled"), Output("rec-cat-pct", "disabled"),
          Input("rec-tope-on", "value"), Input("rec-temp-on", "value"), Input("rec-cat-on", "value"))
def activar(tope_on, temp_on, cat_on):
    return (not tope_on,) * 3 + (not temp_on,) + (not cat_on,) * 2


def _pct(nuevo, base):
    return (nuevo / base - 1) * 100


@callback(Output("rec-kpis", "children"), Output("rec-fig", "figure"), Output("rec-lectura", "children"),
          Output("rec-aportes", "figure"), Output("rec-curva", "figure"), Output("rec-curva-lectura", "children"),
          *CONTROLES)
def actualizar(tope_on, tope, region, perdida, temp_on, temp, cat_on, cat, cat_pct):
    p = simular(bool(tope_on), tope, region, perdida, bool(temp_on), temp, bool(cat_on), cat, cat_pct)
    vb, vs, gb, gs = p.v_base.sum(), p.v3.sum(), p.g_base.sum(), p.g3.sum()
    kpis = [
        dbc.Col(kpi("Ventas 2018", usd(vs), f"base {usd(vb)} · {_pct(vs, vb):+.1f} %"), md=4, className="mb-3"),
        dbc.Col(kpi("Ganancia 2018", usd(gs), f"base {usd(gb)} · {_pct(gs, gb):+.1f} %"), md=4, className="mb-3"),
        dbc.Col(kpi("Margen", f"{gs / vs * 100:.1f} %", f"base {gb / vb * 100:.1f} %"), md=4, className="mb-3"),
    ]

    partes = []
    if tope_on and tope < 80:
        eq = equilibrio(tope, region)
        donde = "toda la empresa" if region == TODAS else region
        partes.append(f"Con un tope de {tope} % en {donde} se recorta el descuento de {eq['lineas']:,} líneas de "
                      f"pedido de 2017.")
        if eq["ganancia"] is None:
            partes.append("Este tope no mejora la ganancia ni aunque no se pierda ningún pedido.")
        elif eq["ganancia"] >= 1:
            partes.append("La ganancia mejora aunque se pierdan todos esos pedidos, porque hoy dejan pérdida.")
        else:
            partes.append(f"La ganancia mejora mientras se pierda menos del {eq['ganancia'] * 100:.0f} % de esos "
                          f"pedidos.")
        if eq["ventas"] is not None and eq["ventas"] < 1:
            partes.append(f"Las ventas no bajan si se pierde menos del {eq['ventas'] * 100:.0f} %.")
    if temp_on and temp:
        partes.append(f"La temporada alta suma {usd(p.v2.sum() - p.v1.sum())} en ventas (supuesto de +{temp} %).")
    if cat_on and cat_pct:
        partes.append(f"Impulsar {cat} suma {usd(p.v3.sum() - p.v2.sum())} en ventas (supuesto de +{cat_pct} %).")
    if not partes:
        partes.append("Active una recomendación para ver cómo cambia la proyección.")
    else:
        partes.append(f"En total, la ganancia de 2018 pasa de {usd(gb)} a {usd(gs)} ({_pct(gs, gb):+.0f} %).")

    curva, c = fig_curva(region, perdida, tope if tope_on else None)
    mejor = c.loc[c.ganancia.idxmax()]
    # Peor caso: se pierden todos los pedidos recortados. Su mejor tope es la conclusión más robusta.
    _, peor = fig_curva(region, 100, None)
    w, real = peor.loc[peor.ganancia.idxmax()], peor[peor.tope == 80].ganancia.iloc[0]
    lectura_curva = insight(
        f"Ganancia que habría tenido 2017 con cada tope, en {'toda la empresa' if region == TODAS else region} y "
        f"suponiendo que se pierde el {perdida} % de los pedidos con descuento recortado. Con este supuesto, el "
        f"tope que deja más ganancia es {mejor.tope:.0f} % ({usd(mejor.ganancia)}). Los topes bajos dependen mucho "
        f"del supuesto, porque recortan descuentos pequeños que hoy sí dejan ganancia. En el peor caso, si se "
        f"perdieran todos esos pedidos, el mejor tope sería {w.tope:.0f} % ({usd(w.ganancia)}"
        + (f", aún por encima de la ganancia real de {usd(real)})." if w.ganancia > real else ")."))
    return kpis, fig_mensual(p), insight(" ".join(partes)), fig_aportes(p), curva, lectura_curva
