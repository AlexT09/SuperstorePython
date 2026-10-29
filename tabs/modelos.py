"""tabs/modelos.py — Modelos implementados y resultados en el periodo de prueba (2017).
Al final, la sección "Entrenamiento del modelo": entrena en vivo cualquiera de los 7 modelos."""
import json
import unicodedata
import warnings
from functools import lru_cache

import dash_bootstrap_components as dbc
import joblib
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from common import (BRAND, BRAND_LIGHT, POS, PROY, REAL, REF, REFERENCIA, aviso_sin_modelo, insight, kpi,
                    load_artifact, load_mensual, mejores_por_modelo, modelo_listo, page_header, soft_card,
                    style_fig, usd)
from model.evaluate import metricas
from model.features import FEATURES, TARGET, a_ventas, agregar_variables, preprocesador
from model.train_model import ANIO_PRUEBA, MODELOS, buscar, con_prefijo, sin_prefijo

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


# ---------------------------------------------------------------- entrenamiento del modelo
@lru_cache(maxsize=1)
def datos_modelo():
    """Tabla del modelo dividida en el tiempo, igual que model/train_model.py: entrenamiento 2015–2016, prueba 2017."""
    datos = agregar_variables(load_mensual()).dropna(subset=FEATURES + [TARGET])
    return datos[datos.Mes.dt.year < ANIO_PRUEBA], datos[datos.Mes.dt.year == ANIO_PRUEBA]


def seccion_entrenamiento():
    control = soft_card("Control de entrenamiento", [
        html.Label("Selecciona un modelo:", className="small fw-bold", htmlFor="ent-sel"),
        dcc.Dropdown(list(MODELOS), "Random Forest", id="ent-sel", clearable=False, className="mb-3"),
        dbc.Button([html.I(className="bi bi-play-fill me-1"), "Entrenar modelo"], id="ent-entrenar",
                   color="success", className="w-100"),
        insight("Busca los mejores hiperparámetros de su rejilla con validación cruzada temporal (2015–2016) y "
                "evalúa una sola vez en 2017, igual que model/train_model.py. Random Forest y XGBoost tardan "
                "unos segundos."),
        html.Hr(),
        dbc.Button([html.I(className="bi bi-floppy me-1"), "Guardar modelo"], id="ent-guardar",
                   color="secondary", outline=True, className="w-100"),
        insight("Descarga el modelo entrenado como archivo .pkl (Pipeline de scikit-learn, se abre con joblib)."),
        html.Div(id="ent-aviso"),
        dcc.Store(id="ent-modelo"),
        dcc.Download(id="ent-descarga"),
    ], "blue")
    resultados = soft_card("Resultados del modelo", dcc.Loading(
        html.Div(insight("Elige un modelo y presiona «Entrenar modelo»."), id="ent-resultado"),
        type="dot", color=PROY), "lavender")
    return html.Div([
        html.H4("Entrenamiento del modelo: entrena cualquiera de los 7 modelos", className="section-title"),
        dbc.Row([dbc.Col(control, lg=4, className="mb-3"), dbc.Col(resultados, lg=8, className="mb-3")]),
    ])


def fig_entrenamiento(train, test, pred_train, pred_test):
    """Real vs. predicho del modelo recién entrenado, en el total de la empresa por mes."""
    t = pd.concat([train.assign(conjunto="Entrenamiento", pred=pred_train),
                   test.assign(conjunto="Prueba", pred=pred_test)])
    t = t.groupby(["Mes", "conjunto"], as_index=False)[["Sales", "pred"]].sum()
    fig = go.Figure()
    fig.add_vrect(x0=test.Mes.min(), x1=test.Mes.max(), fillcolor=BRAND_LIGHT, opacity=0.7, line_width=0,
                  layer="below", annotation_text="Prueba (2017)", annotation_position="top left")
    fig.add_trace(go.Scatter(x=t.Mes, y=t.Sales, name="Real", mode="lines+markers",
                             line=dict(color=REAL, width=2), marker_size=5))
    fig.add_trace(go.Scatter(x=t.Mes, y=t.pred, name="Predicho", mode="lines+markers",
                             line=dict(color=PROY, width=2.5), marker_size=5))
    fig.update_yaxes(title="Ventas del mes (USD)", tickprefix="$", tickformat=",.0f")
    return style_fig(fig, 320)


@callback(Output("ent-resultado", "children"), Output("ent-modelo", "data"),
          Input("ent-entrenar", "n_clicks"), State("ent-sel", "value"), prevent_initial_call=True)
def entrenar(_, nombre):
    """Entrena el modelo elegido al presionar el botón."""
    warnings.filterwarnings("ignore", category=UserWarning)
    train, test = datos_modelo()
    est, grid = MODELOS[nombre]
    s, seg = buscar(clone(est), grid, train[FEATURES], train[TARGET])
    pred_test = a_ventas(s.predict(test[FEATURES]), test)
    pred_train = a_ventas(s.predict(train[FEATURES]), train)
    m, ref = metricas(test.Sales, pred_test), metricas(test.Sales, test.lag_12)
    mejora = (1 - m["MAE"] / ref["MAE"]) * 100
    params = sin_prefijo(s.best_params_)
    contenido = [
        dbc.Row([
            dbc.Col(kpi("MAE en 2017", usd(m["MAE"])), md=3, className="mb-3"),
            dbc.Col(kpi("RMSE en 2017", usd(m["RMSE"])), md=3, className="mb-3"),
            dbc.Col(kpi("R² en 2017", f"{m['R2']:.2f}"), md=3, className="mb-3"),
            dbc.Col(kpi("Frente a la referencia", f"{mejora:+.0f} %",
                        f"menos error que la referencia ({usd(ref['MAE'])})"), md=3, className="mb-3"),
        ]),
        html.H6("Real vs. predicho (total de la empresa por mes)"),
        dcc.Graph(figure=fig_entrenamiento(train, test, pred_train, pred_test), config=CFG),
        html.H6("Hiperparámetros elegidos", className="mt-2"),
        html.Ul([
            html.Li(f"Modelo: {nombre}"),
            html.Li(f"Combinaciones probadas: {len(s.cv_results_['params'])} en {seg:.1f} segundos"),
            html.Li("Mejores hiperparámetros: " + ", ".join(f"{k}={_fmt(v)}" for k, v in params.items())),
            html.Li(f"Error de validación cruzada (MAE del índice): {-s.best_score_:.4f}"),
        ], className="mb-0"),
    ]
    return contenido, {"nombre": nombre, "params": params}


@callback(Output("ent-descarga", "data"), Output("ent-aviso", "children"),
          Input("ent-guardar", "n_clicks"), State("ent-modelo", "data"), prevent_initial_call=True)
def guardar(_, entrenado):
    """Descarga el modelo entrenado como .pkl."""
    if not entrenado:
        return no_update, dbc.Alert("Primero entrena un modelo.", color="warning", className="mt-3 mb-0 py-2")
    # Mismo modelo, hiperparámetros y datos que al entrenar: con la semilla fija el ajuste se repite igual
    nombre = entrenado["nombre"]
    train, _ = datos_modelo()
    pipe = Pipeline([("pre", preprocesador()), ("modelo", clone(MODELOS[nombre][0]))])
    pipe.set_params(**con_prefijo(entrenado["params"]))
    pipe.fit(train[FEATURES], train[TARGET])
    slug = unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode().lower().replace(" ", "_")
    archivo = f"modelo_{slug}.pkl"
    return dcc.send_bytes(lambda b: joblib.dump(pipe, b), archivo), dbc.Alert(
        f"Modelo guardado como {archivo}.", color="success", className="mt-3 mb-0 py-2")


def layout():
    if not modelo_listo():
        return html.Div([page_header("Modelos y resultados"), aviso_sin_modelo(), seccion_entrenamiento()])
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
    rec = resumen["prueba_recursiva"]
    # Si otro modelo tuvo menos error en 2017, explicar por qué no se eligió
    menor, elegido = res.iloc[0], res.set_index("modelo").loc[best]
    nota_menor = "" if menor.modelo == best else (
        f" {menor.modelo} tuvo el menor MAE en 2017 ({usd(menor.MAE)}) y no se eligió por dos razones: elegir "
        f"mirando 2017 haría que esa prueba dejara de ser independiente (en validación cruzada su error fue "
        f"{menor.cv_MAE_indice:.3f}, contra {elegido.cv_MAE_indice:.3f} de {best}), y su total del año quedó a "
        f"{menor.total_desvio_total_pct:+.1f} % del real: subestima el año, mientras que {best} quedó a "
        f"{elegido.total_desvio_total_pct:+.1f} %.")
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
        html.Div(soft_card("Ranking de modelos (MAE en 2017)", [
            dcc.Graph(figure=fig_ranking(res, ref, best), config=CFG),
            insight(f"Todos los modelos con MAE a la izquierda de la línea superan a la referencia. Las "
                    f"diferencias entre ellos son pequeñas (de {usd(res.MAE.min())} a {usd(res.MAE.max())}): "
                    f"con 288 filas de entrenamiento, la señal disponible es sobre todo la estacionalidad, "
                    f"y todos la capturan. {best} (en verde) se eligió por su error de validación cruzada, "
                    f"antes de mirar 2017." + nota_menor),
        ], "blue", height_full=False), className="mt-3"),
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
        seccion_entrenamiento(),
    ])


@callback(Output("modelos-ajuste", "figure"), Input("modelos-sel", "value"))
def actualizar_ajuste(modelo):
    return fig_ajuste(modelo)
