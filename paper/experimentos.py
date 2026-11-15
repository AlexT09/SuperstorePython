"""
paper/experimentos.py — Reproduce todas las tablas, pruebas estadísticas y figuras del artículo.

Modelo propuesto: perfil estacional global contraído (paper/perfil_estacional.py).
Referencias: ingenuo estacional y los modelos del curso (los 7 de model/train_model.py, con sus
mismas rejillas, más regresión bayesiana).

Protocolos:
  B (primario)   Origen móvil: 24 orígenes mensuales (2016-01 a 2017-12). En cada uno, todos los
                 modelos se reentrenan con los meses anteriores (ventana creciente desde 2015-01) y
                 reajustan sus hiperparámetros con validación cruzada temporal anidada; se
                 pronostica el mes siguiente de las 12 series.
  A (secundario) Partición del proyecto: entrenamiento 2015–2016, prueba 2017, a un paso y con
                 proyección recursiva a 12 meses.

Ejecutar desde la raíz del repositorio:
    python paper/experimentos.py            # calcula todo (~6 min) y genera las figuras
    python paper/experimentos.py --figuras  # solo regenera las figuras desde resultados/cache.pkl

Salidas: paper/resultados/*.csv, paper/resultados/numeros.json y paper/figures/*.pdf
"""
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
PAPER = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PAPER))
from model.features import CATEGORICAS, FEATURES, NUMERICAS, SEED, TARGET, agregar_variables  # noqa: E402
from model.train_model import CV, MODELOS, REFERENCIA, con_prefijo, sin_prefijo  # noqa: E402
from perfil_estacional import REJILLA, PerfilEstacional  # noqa: E402

RES, FIG = PAPER / "resultados", PAPER / "figures"
PROPUESTO = "Perfil estacional"
BAYES = "Regresión bayesiana"
RF = "Random Forest"
ALPHA = 0.05
ROPE_PCT = 0.01                      # región de equivalencia práctica: ±1 % de la venta media previa al primer origen
N_BOOT = 10000
BLOQUE = 3                           # longitud del bloque (meses) del bootstrap
ORIGENES = pd.date_range("2016-01-01", "2017-12-01", freq="MS")
VENTANAS = [12, 18, 24]              # meses máximos de entrenamiento (sensibilidad)
MAE_SCORE = "neg_mean_absolute_error"

ORDEN = [REFERENCIA, "Ridge", "Lasso", BAYES, "KNN", "Árbol de decisión", RF, "XGBoost", "SVR", PROPUESTO]
EN = {REFERENCIA: "Seasonal naive", "Ridge": "Ridge", "Lasso": "Lasso", BAYES: "Bayesian ridge", "KNN": "KNN",
      "Árbol de decisión": "Decision tree", RF: "Random forest", "XGBoost": "XGBoost", "SVR": "SVR",
      PROPUESTO: "Seasonal profile (proposed)"}

# Ablación: nombre -> (estimador, rejilla, usa nivel estático)
COMPLETO = "Modelo completo"
ABLACION = {
    COMPLETO: (PerfilEstacional(), REJILLA, False),
    "Sin perfil estacional (constante)": (PerfilEstacional(perfil="ninguno"), REJILLA, False),
    "Sin rezago de la serie (w = 0)": (PerfilEstacional(), {"w": [0.0], "estadistico": REJILLA["estadistico"]}, False),
    "Sin agrupar series (perfil por serie)": (PerfilEstacional(perfil="serie"), REJILLA, False),
    "Sin nivel móvil (nivel estático)": (PerfilEstacional(), REJILLA, True),
    "Mediana en lugar de media": (PerfilEstacional(), {"w": REJILLA["w"], "estadistico": ["mediana"]}, False),
}
EN_AB = {COMPLETO: "Full model", "Sin perfil estacional (constante)": "No seasonal profile",
         "Sin rezago de la serie (w = 0)": "No series lag (w = 0)",
         "Sin agrupar series (perfil por serie)": "No pooling (per-series profile)",
         "Sin nivel móvil (nivel estático)": "No rolling level (static level)",
         "Mediana en lugar de media": "Median instead of mean"}


# ----------------------------------------------------------------------------- utilidades
def a_usd(indice, filas):
    """Del índice a dólares; nunca negativo."""
    return np.clip(np.asarray(indice) * filas.prom_12m.to_numpy(), 0, None)


def buscar(est, grid, tr, cv):
    """Búsqueda en rejilla de un modelo de referencia dentro del Pipeline del proyecto."""
    pipe = Pipeline([("pre", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
                                                ("num", StandardScaler(), NUMERICAS)])), ("modelo", est)])
    s = GridSearchCV(pipe, con_prefijo(grid), cv=cv, scoring=MAE_SCORE, n_jobs=-1)
    t0 = time.perf_counter()
    s.fit(tr[FEATURES], tr[TARGET])
    return s, time.perf_counter() - t0


def buscar_perfil(est, grid, tr, cv):
    """Búsqueda en rejilla del modelo propuesto (o de una variante); recibe la tabla directamente."""
    s = GridSearchCV(clone(est), grid, cv=cv, scoring=MAE_SCORE, n_jobs=1)
    t0 = time.perf_counter()
    s.fit(tr, tr[TARGET])
    return s, time.perf_counter() - t0


def tiempos(est, X_tr, y_tr, X_te, rep_fit=3, rep_inf=10):
    """Mediana del tiempo de reajuste y de predicción de la configuración elegida (ms, un hilo)."""
    tf, ti = [], []
    for _ in range(rep_fit):
        m = clone(est)
        t0 = time.perf_counter(); m.fit(X_tr, y_tr); tf.append(time.perf_counter() - t0)
    for _ in range(rep_inf):
        t0 = time.perf_counter(); est.predict(X_te); ti.append(time.perf_counter() - t0)
    return np.median(tf) * 1e3, np.median(ti) * 1e3


def cv_interna(n_meses):
    """Pliegues de 3 meses, hasta 4, dejando al menos 6 meses de entrenamiento en el primero."""
    return TimeSeriesSplit(n_splits=min(4, (n_meses - 6) // 3), test_size=36)


def nivel_estatico(d, panel, origen):
    """Variante de ablación: el nivel es la media de toda la historia previa al origen, no la de 12 meses."""
    niv = panel[panel.Mes < origen].groupby(["Category", "Region"]).Sales.mean().rename("niv")
    d = d.join(niv, on=["Category", "Region"])
    return d.assign(prom_12m=d.niv, indice=d.Sales / d.niv, r_lag_12=d.lag_12 / d.niv)


def metricas(real, pred, escala):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    e = real - pred
    return {"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()), "WAPE": np.abs(e).sum() / real.sum() * 100,
            "MASE": np.abs(e).mean() / escala, "R2": r2_score(real, pred)}


def desvio_total(filas, pred):
    return (np.sum(pred) / filas.Sales.sum() - 1) * 100


def mae_mensual(filas, pred):
    """MAE de cada mes (promedio sobre las 12 series): los bloques temporales de la evaluación."""
    return filas.assign(e=np.abs(filas.Sales.to_numpy() - np.asarray(pred))).groupby("Mes").e.mean().to_numpy()


def recursivo(predecir, panel_hist, meses):
    """Proyección recursiva: cada mes proyectado pasa a ser historia del siguiente."""
    hist = panel_hist[["Category", "Region", "Mes", "Sales"]].copy()
    series = hist[["Category", "Region"]].drop_duplicates()
    for mes in meses:
        hist = pd.concat([hist, series.assign(Mes=mes, Sales=np.nan)], ignore_index=True)
        filas = agregar_variables(hist)
        filas = filas[filas.Mes == mes]
        p = predecir(filas)
        hist = hist[hist.Mes != mes]
        hist = pd.concat([hist, filas[["Category", "Region", "Mes"]].assign(Sales=p)], ignore_index=True)
    return hist[hist.Mes.isin(meses)].sort_values(["Mes", "Category", "Region"]).Sales.to_numpy()


# ----------------------------------------------------------------------------- pruebas estadísticas
def dm_hln(d, h=1):
    """Diebold–Mariano con corrección de Harvey–Leybourne–Newbold sobre el diferencial de pérdida d_t."""
    d = np.asarray(d, float)
    T, dbar = d.size, d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[:T - k] - dbar)) / T for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / T
    stat = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T) * dbar / np.sqrt(var)
    return stat, 2 * stats.t.sf(abs(stat), df=T - 1)


def giacomini_white(d):
    """Giacomini–White (capacidad predictiva condicional, h = 1) con instrumentos (1, d_{t-1})."""
    d = np.asarray(d, float)
    H = np.column_stack([np.ones(d.size - 1), d[:-1]])
    Z = H * d[1:, None]
    n, zbar = Z.shape[0], Z.mean(axis=0)
    stat = float(n * zbar @ np.linalg.pinv(Z.T @ Z / n) @ zbar)
    return stat, stats.chi2.sf(stat, df=H.shape[1])


def bayes_rope(x, rope, rho):
    """t correlacionada bayesiana (Corani y Benavoli): x = puntaje referencia − propuesto (positivo = mejora).

    Devuelve P(propuesto mejor), P(equivalentes dentro de la ROPE), P(propuesto peor).
    """
    x = np.asarray(x, float)
    n, m, v = x.size, x.mean(), x.var(ddof=1)
    post = stats.t(df=n - 1, loc=m, scale=np.sqrt((1 / n + rho / (1 - rho)) * v))
    return 1 - post.cdf(rope), post.cdf(rope) - post.cdf(-rope), post.cdf(-rope)


def ic_bootstrap(d, rng):
    """IC del 95 % de la media de d_t con bootstrap de bloques móviles circulares (dependencia temporal)."""
    d = np.asarray(d, float)
    T, nb = d.size, int(np.ceil(d.size / BLOQUE))
    ini = rng.integers(0, T, size=(N_BOOT, nb))
    idx = (ini[:, :, None] + np.arange(BLOQUE)[None, None, :]) % T
    return np.percentile(d[idx.reshape(N_BOOT, -1)[:, :T]].mean(axis=1), [2.5, 97.5])


def contrastar(nombre, err_prop, err_base, filas, rope, rng):
    """Todas las pruebas del modelo propuesto frente a una referencia (o del modelo completo frente a una variante)."""
    f = filas.assign(d=np.abs(err_prop) - np.abs(err_base), d2=err_prop ** 2 - err_base ** 2)
    d = f.groupby("Mes").d.mean().to_numpy()     # diferencial de pérdida absoluta por mes; negativo = propuesto mejor
    d2 = f.groupby("Mes").d2.mean().to_numpy()
    dm, p_dm = dm_hln(d)
    dm2, p_dm2 = dm_hln(d2)
    gw, p_gw = giacomini_white(d)
    mejor, igual, peor = bayes_rope(-d, rope, rho=1 / d.size)
    lo, hi = ic_bootstrap(d, rng)
    return {"baseline": nombre, "dif_MAE": d.mean(), "ic_lo": lo, "ic_hi": hi, "d_cohen": d.mean() / d.std(ddof=1),
            "DM": dm, "p_DM": p_dm, "DM_sq": dm2, "p_DM_sq": p_dm2, "GW": gw, "p_GW": p_gw,
            "P_mejor": mejor, "P_rope": igual, "P_peor": peor, "p_wilcoxon": stats.wilcoxon(d).pvalue,
            "shapiro_p": stats.shapiro(d).pvalue, "meses_mejor": int((d < 0).sum())}


def holm(df, cols):
    for c in cols:
        df[c + "_holm"] = multipletests(df[c], alpha=ALPHA, method="holm")[1]
    return df


# ----------------------------------------------------------------------------- cálculo
def describir(crudo, lineas, panel, datos):
    def desc(s):
        q1, q3 = s.quantile([.25, .75])
        return {"n": int(s.size), "media": s.mean(), "mediana": s.median(), "sd": s.std(), "min": s.min(),
                "max": s.max(), "asimetria": s.skew(), "faltantes": int(s.isna().sum()),
                "atipicos_iqr": int(((s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))).sum())}
    anual = panel.groupby(panel.Mes.dt.year).Sales.sum()
    por_mes = panel.groupby(panel.Mes.dt.month).Sales.sum()
    return {
        "filas_crudo": len(crudo), "columnas_crudo": crudo.shape[1], "duplicados": int(crudo.duplicated().sum()),
        "faltantes_crudo": int(crudo.isna().sum().sum()),
        "fecha_min": str(lineas.Order_Date.min().date()), "fecha_max": str(lineas.Order_Date.max().date()),
        "linea": {c: desc(lineas[c]) for c in ["Sales", "Quantity", "Discount", "Profit"]},
        "panel_sales": desc(panel.Sales), "indice": desc(datos[TARGET]), "panel_ceros": int((panel.Sales == 0).sum()),
        "anual": anual.round(0).to_dict(), "crec_anual_pct": (anual.pct_change() * 100).round(1).dropna().to_dict(),
        "cuota_categoria": (panel.groupby("Category").Sales.sum() / panel.Sales.sum() * 100).round(1).to_dict(),
        "cuota_region": (panel.groupby("Region").Sales.sum() / panel.Sales.sum() * 100).round(1).to_dict(),
        "cuota_mes": (por_mes / por_mes.sum() * 100).round(1).to_dict(),
        "corr_linea": lineas[["Sales", "Quantity", "Discount", "Profit"]].corr(method="spearman").Sales.round(3).to_dict(),
        "corr_indice": datos[[TARGET] + NUMERICAS].corr(method="spearman")[TARGET].round(3).to_dict(),
    }


def modelo_final(datos, panel):
    """Modelo reentrenado con 2015–2017 y proyección recursiva de 2018.

    Con los tres años, la validación cruzada puede preferir la mediana (menor MAE, pero sesgo de nivel:
    ver ablación). Se reportan las dos proyecciones: la de la configuración elegida y la del perfil de medias.
    """
    meses18 = pd.date_range("2018-01-01", periods=12, freq="MS")
    out = {"ventas_2017": float(panel[panel.Mes.dt.year == 2017].Sales.sum())}
    for clave, grid in [("elegido_vc", REJILLA), ("media", {"w": REJILLA["w"], "estadistico": ["media"]})]:
        s, _ = buscar_perfil(PerfilEstacional(), grid, datos, CV)
        m = s.best_estimator_
        proy = recursivo(lambda f, m=m: a_usd(m.predict(f), f), panel, meses18)
        out[clave] = {"hiperparametros": s.best_params_, "cv_MAE": float(-s.best_score_),
                      "proyeccion_2018": float(proy.sum()),
                      "perfil_ene_dic": [float(v) for v in np.r_[m.perfil_[1:], m.perfil_[0]]]}
        perfil = m.perfil_
    return out, perfil      # perfil de medias (el que se dibuja)


def calcular():
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(SEED)
    crudo = pd.read_csv(ROOT / "data" / "raw" / "Sample - Superstore.csv", encoding="latin1")
    lineas = pd.read_csv(ROOT / "data" / "superstore_transformado.csv", parse_dates=["Order_Date"])
    panel = pd.read_csv(ROOT / "data" / "ventas_mensuales.csv", parse_dates=["Mes"])
    datos = agregar_variables(panel).dropna(subset=FEATURES + [TARGET])
    N = {"eda": describir(crudo, lineas, panel, datos)}

    modelos = dict(MODELOS)
    modelos[BAYES] = (BayesianRidge(), {"alpha_1": [1e-6, 1e-3, 1.0], "lambda_1": [1e-6, 1e-3, 1.0]})
    configs = [(w, e) for e in REJILLA["estadistico"] for w in REJILLA["w"] + [0.6, 0.8, 1.0]]

    # ======================= Protocolo B: origen móvil =======================
    filas, costos, elegidos = [], [], []
    for origen in ORIGENES:
        tr, te = datos[datos.Mes < origen], datos[datos.Mes == origen]
        cv = cv_interna(tr.Mes.nunique())
        out = te[["Category", "Region", "Mes", "Sales"]].copy()
        out[REFERENCIA] = te.lag_12.to_numpy()
        for nombre, (est, grid) in modelos.items():
            s, seg = buscar(clone(est), grid, tr, cv)
            out[nombre] = a_usd(s.predict(te[FEATURES]), te)
            t_fit, t_inf = tiempos(s.best_estimator_, tr[FEATURES], tr[TARGET], te[FEATURES])
            costos.append({"modelo": nombre, "busqueda_s": seg, "ajuste_ms": t_fit, "inferencia_ms": t_inf,
                           "tamano_kB": len(pickle.dumps(s.best_estimator_)) / 1024})
            elegidos.append({"origen": origen, "modelo": nombre, **sin_prefijo(s.best_params_)})
        for nombre, (est, grid, estatico) in ABLACION.items():
            tr2, te2 = (nivel_estatico(tr, panel, origen), nivel_estatico(te, panel, origen)) if estatico else (tr, te)
            s, seg = buscar_perfil(est, grid, tr2, cv)
            col = PROPUESTO if nombre == COMPLETO else "abl|" + nombre
            out[col] = a_usd(s.predict(te2), te2)
            if nombre == COMPLETO:
                t_fit, t_inf = tiempos(s.best_estimator_, tr, tr[TARGET], te)
                costos.append({"modelo": PROPUESTO, "busqueda_s": seg, "ajuste_ms": t_fit, "inferencia_ms": t_inf,
                               "tamano_kB": len(pickle.dumps(s.best_estimator_)) / 1024})
                elegidos.append({"origen": origen, "modelo": PROPUESTO, **s.best_params_, "cv_MAE": -s.best_score_})
        for w, e in configs:   # sensibilidad: configuración fija, sin búsqueda
            out[f"fijo|{e}|{w}"] = a_usd(PerfilEstacional(w, e).fit(tr, tr[TARGET]).predict(te), te)
        for k in VENTANAS:     # sensibilidad: ventana de entrenamiento limitada a los últimos k meses
            trk = tr[tr.Mes >= origen - pd.DateOffset(months=k)]
            s, _ = buscar_perfil(PerfilEstacional(), REJILLA, trk, cv_interna(trk.Mes.nunique()))
            out[f"ventana|{k}"] = a_usd(s.predict(te), te)
        filas.append(out)
        print(f"  origen {origen:%Y-%m} · {tr.Mes.nunique()} meses de entrenamiento", flush=True)

    p = pd.concat(filas, ignore_index=True)
    real = p.Sales.to_numpy()
    previo = datos[datos.Mes < ORIGENES[0]]
    escala = float(np.abs(previo.Sales - previo.lag_12).mean())     # MAE del ingenuo estacional antes del primer origen
    rope = ROPE_PCT * float(previo.Sales.mean())
    es2016 = (p.Mes.dt.year == 2016).to_numpy()
    costo = pd.DataFrame(costos).groupby("modelo").median()
    elegidos = pd.DataFrame(elegidos)

    def fila(col):
        e = real - p[col].to_numpy()
        return {**metricas(real, p[col], escala), "MAE_sd": mae_mensual(p, p[col]).std(ddof=1),
                "desvio_total_pct": desvio_total(p, p[col]), "MAE_2016": np.abs(e[es2016]).mean(),
                "MAE_2017": np.abs(e[~es2016]).mean()}

    principal = pd.DataFrame([{"modelo": n, **fila(n)} for n in ORDEN]).set_index("modelo").join(costo)
    err = {n: real - p[n].to_numpy() for n in ORDEN}
    sig = holm(pd.DataFrame([contrastar(b, err[PROPUESTO], err[b], p, rope, rng) for b in ORDEN if b != PROPUESTO]),
               ["p_DM", "p_DM_sq", "p_GW", "p_wilcoxon"]).set_index("baseline")

    # --- Ablación (mismo protocolo, mismas pruebas)
    variantes = [v for v in ABLACION if v != COMPLETO]
    abl = pd.DataFrame([{"variante": COMPLETO, **fila(PROPUESTO)}] +
                       [{"variante": v, **fila("abl|" + v)} for v in variantes]).set_index("variante")
    abl_sig = holm(pd.DataFrame([contrastar(v, err[PROPUESTO], real - p["abl|" + v].to_numpy(), p, rope, rng)
                                 for v in variantes]), ["p_DM", "p_GW"]).set_index("baseline")
    ablacion = abl.join(abl_sig[["dif_MAE", "ic_lo", "ic_hi", "DM", "p_DM", "p_DM_holm", "p_GW", "p_GW_holm",
                                 "P_mejor", "P_rope", "P_peor"]])
    abl_ic = {v: ic_bootstrap(mae_mensual(p, p[PROPUESTO if v == COMPLETO else "abl|" + v]), rng) for v in ABLACION}

    # --- Sensibilidad
    sens_w = pd.DataFrame([{"w": w, "estadistico": e, **{k: v for k, v in fila(f"fijo|{e}|{w}").items()
                                                          if k in ("MAE", "RMSE", "desvio_total_pct")}} for w, e in configs])
    sens_v = pd.DataFrame([{"ventana": str(k), **fila(f"ventana|{k}")} for k in VENTANAS] +
                          [{"ventana": "todos", **fila(PROPUESTO)}])[["ventana", "MAE", "MAE_sd", "desvio_total_pct"]]
    w_origen = elegidos[elegidos.modelo == PROPUESTO][["origen", "w", "estadistico", "cv_MAE"]]

    # --- Errores e interpretación (pronósticos de origen móvil del modelo propuesto)
    t = p[["Category", "Region", "Mes", "Sales"]].assign(pred=p[PROPUESTO], ref=p[REFERENCIA])
    t["e"], t["ae"], t["ae_ref"] = t.Sales - t.pred, (t.Sales - t.pred).abs(), (t.Sales - t.ref).abs()
    por_serie = t.groupby(["Category", "Region"]).agg(venta_media=("Sales", "mean"), MAE=("ae", "mean"),
                                                      MAE_ref=("ae_ref", "mean"), sesgo=("e", "mean")).reset_index()
    por_serie["WAPE"] = por_serie.MAE / por_serie.venta_media * 100
    por_serie["WAPE_ref"] = por_serie.MAE_ref / por_serie.venta_media * 100
    por_mes = t.groupby("Mes").agg(real=("Sales", "sum"), pred=("pred", "sum"), ref=("ref", "sum"),
                                   MAE=("ae", "mean"), MAE_ref=("ae_ref", "mean")).reset_index()
    t["tercil"] = pd.qcut(t.Sales, 3, labels=["bajo", "medio", "alto"])
    por_tercil = t.groupby("tercil", observed=True).agg(venta_media=("Sales", "mean"), MAE=("ae", "mean"),
                                                        sesgo=("e", "mean"), MAE_ref=("ae_ref", "mean")).reset_index()
    peores = t.nlargest(5, "ae")[["Category", "Region", "Mes", "Sales", "pred", "e"]]
    tot_abs = lambda c: float((por_mes[c] - por_mes.real).abs().sum() / por_mes.real.sum() * 100)  # noqa: E731
    sesgo_tot = {n: p.assign(x=p[n]).groupby("Mes").x.sum().to_numpy() - por_mes.real.to_numpy() for n in ORDEN}
    sesgo = pd.DataFrame([{"modelo": n, "sesgo_medio": v.mean(), **dict(zip(["lo", "hi"], ic_bootstrap(v, rng))),
                           "WAPE_total": float(np.abs(v).sum() / por_mes.real.sum() * 100)}
                          for n, v in sesgo_tot.items()]).set_index("modelo")
    N["errores"] = {"series_mejor_que_ref": int((por_serie.MAE < por_serie.MAE_ref).sum()),
                    "meses_mejor_que_ref": int((por_mes.MAE < por_mes.MAE_ref).sum()),
                    "residuo_media": float(t.e.mean()), "residuo_asimetria": float(t.e.skew()),
                    "WAPE_total": tot_abs("pred"), "WAPE_total_ref": tot_abs("ref")}

    # ======================= Protocolo A: partición del proyecto =======================
    train, test = datos[datos.Mes.dt.year < 2017], datos[datos.Mes.dt.year == 2017]
    escala_a = float(np.abs(train.Sales - train.lag_12).mean())
    meses17 = pd.date_range("2017-01-01", periods=12, freq="MS")
    hist16 = panel[panel.Mes.dt.year < 2017]
    real_a = test.Sales.to_numpy()
    pa, ra, cv_a = {REFERENCIA: test.lag_12.to_numpy()}, {REFERENCIA: test.lag_12.to_numpy()}, {}
    for nombre, (est, grid) in modelos.items():
        s, _ = buscar(clone(est), grid, train, CV)
        pa[nombre] = a_usd(s.predict(test[FEATURES]), test)
        ra[nombre] = recursivo(lambda f, s=s: a_usd(s.predict(f[FEATURES]), f), hist16, meses17)
        cv_a[nombre] = -s.best_score_
    s_a, _ = buscar_perfil(PerfilEstacional(), REJILLA, train, CV)
    pa[PROPUESTO] = a_usd(s_a.predict(test), test)
    ra[PROPUESTO] = recursivo(lambda f: a_usd(s_a.predict(f), f), hist16, meses17)
    cv_a[PROPUESTO] = -s_a.best_score_
    protocolo_a = pd.DataFrame([{"modelo": n, **metricas(real_a, pa[n], escala_a),
                                 "MAE_sd": mae_mensual(test, pa[n]).std(ddof=1),
                                 "desvio_total_pct": desvio_total(test, pa[n]),
                                 "rec_MAE": np.abs(real_a - ra[n]).mean(), "rec_desvio_total_pct": desvio_total(test, ra[n]),
                                 "cv_MAE_indice": cv_a.get(n, np.nan)} for n in ORDEN]).set_index("modelo")
    sig_a = holm(pd.DataFrame([contrastar(b, real_a - pa[PROPUESTO], real_a - pa[b], test, ROPE_PCT * float(train.Sales.mean()), rng)
                               for b in ORDEN if b != PROPUESTO]), ["p_DM", "p_GW"]).set_index("baseline")
    N["protocolo_a"] = {"hiperparametros": s_a.best_params_,
                        "total_WAPE_1paso": float(np.abs(test.assign(x=pa[PROPUESTO]).groupby("Mes").x.sum().to_numpy()
                                                         - test.groupby("Mes").Sales.sum().to_numpy()).sum() / real_a.sum() * 100),
                        "total_WAPE_rec": float(np.abs(test.assign(x=ra[PROPUESTO]).groupby("Mes").x.sum().to_numpy()
                                                       - test.groupby("Mes").Sales.sum().to_numpy()).sum() / real_a.sum() * 100)}
    rec17 = test[["Mes", "Sales"]].assign(rec=ra[PROPUESTO]).groupby("Mes").sum().reset_index()

    # ======================= Modelo final y proyección 2018 =======================
    N["final"], perfil_final = modelo_final(datos, panel)
    N["protocolo_b"] = {"n_obs": len(p), "n_origenes": len(ORIGENES), "escala_mase": escala, "rope_usd": rope,
                        "venta_media_evaluacion": float(real.mean())}

    c = dict(N=N, principal=principal, sig=sig, ablacion=ablacion, abl_ic=abl_ic, sens_w=sens_w, sens_v=sens_v,
             w_origen=w_origen, por_serie=por_serie, por_mes=por_mes, por_tercil=por_tercil, peores=peores,
             sesgo=sesgo, protocolo_a=protocolo_a, sig_a=sig_a, rec17=rec17, elegidos=elegidos,
             p=p, panel=panel, datos=datos, perfil=perfil_final)
    RES.mkdir(exist_ok=True)
    with open(RES / "cache.pkl", "wb") as f:
        pickle.dump(c, f)
    for k in ["principal", "sig", "ablacion", "sens_w", "sens_v", "w_origen", "por_serie", "por_mes", "por_tercil",
              "peores", "sesgo", "protocolo_a", "sig_a", "elegidos"]:
        c[k].round(4).to_csv(RES / f"{k}.csv")
    p.round(2).to_csv(RES / "predicciones_origen_movil.csv", index=False)
    (RES / "numeros.json").write_text(json.dumps(N, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return c


# ----------------------------------------------------------------------------- figuras
def figuras(c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Latin Modern Sans", "DejaVu Sans"],
        "font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 12, "lines.linewidth": 2.0, "axes.grid": True, "grid.alpha": 0.3,
        "pdf.fonttype": 42, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
    })
    AZUL, AMBAR, GRIS, CLARO = "#1F3A5F", "#E0A526", "#7A7A7A", "#C9D6E3"
    FIG.mkdir(exist_ok=True)
    panel, p, d = c["panel"], c["p"], c["datos"]
    INICIALES = list("JFMAMJJASOND")

    # 1. EDA: serie total y dispersión por categoría
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.4), gridspec_kw={"width_ratios": [1.5, 1]})
    tot = panel.groupby("Mes").Sales.sum()
    ax[0].plot(tot.index, tot.values / 1000, color=AZUL)
    ax[0].axvspan(pd.Timestamp("2016-01-01"), pd.Timestamp("2017-12-31"), color=AMBAR, alpha=0.25, lw=0)
    ax[0].axvspan(pd.Timestamp("2014-01-01"), pd.Timestamp("2014-12-31"), color=GRIS, alpha=0.15, lw=0)
    for x, txt in [("2014-07-01", "history"), ("2015-07-01", "initial\ntraining"), ("2017-01-01", "rolling-origin\nevaluation")]:
        ax[0].text(pd.Timestamp(x), 135, txt, ha="center", va="top", fontsize=9)
    ax[0].set_ylim(0, 138)
    ax[0].set_xlabel("Month"); ax[0].set_ylabel("Total sales (thousand USD)")
    ax[0].set_xticks(pd.to_datetime(["2014-01-01", "2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01"]))
    ax[0].set_xticklabels(["2014", "2015", "2016", "2017", "2018"])
    cats = sorted(panel.Category.unique())
    ax[1].boxplot([panel[panel.Category == g].Sales / 1000 for g in cats], widths=0.6,
                  medianprops={"color": AMBAR, "linewidth": 2}, flierprops={"marker": "o", "markersize": 4})
    ax[1].set_xticks(range(1, 4)); ax[1].set_xticklabels(["Furniture", "\nOffice supplies", "Technology"], fontsize=10)
    ax[1].set_ylabel("Monthly sales per series\n(thousand USD)")
    fig.tight_layout(); fig.savefig(FIG / "eda_overview.pdf"); plt.close(fig)

    # 2. EDA: estacionalidad del índice y relación con el mismo mes del año anterior
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.2))
    ax[0].boxplot([d[d.Mes.dt.month == m].indice for m in range(1, 13)], widths=0.6,
                  medianprops={"color": AMBAR, "linewidth": 2}, flierprops={"marker": "o", "markersize": 3})
    ax[0].axhline(1, color=GRIS, lw=1, ls="--")
    ax[0].set_xticks(range(1, 13)); ax[0].set_xticklabels(INICIALES, fontsize=10)
    ax[0].set_xlabel("Month of the year"); ax[0].set_ylabel("Sales index (ratio)")
    ax[1].scatter(d.r_lag_12, d.indice, s=12, color=AZUL, alpha=0.5, lw=0)
    ax[1].plot([0, 4], [0, 4], color=GRIS, lw=1, ls="--")
    ax[1].set_xlim(0, 4.2); ax[1].set_ylim(0, 4.2)
    ax[1].set_xlabel("Same month of previous\nyear (ratio)"); ax[1].set_ylabel("Sales index (ratio)")
    fig.tight_layout(); fig.savefig(FIG / "eda_index.pdf"); plt.close(fig)

    # 3. Diagrama del método (ciclo: panel -> índice -> perfil -> ventas -> panel)
    fig, ax = plt.subplots(figsize=(6.5, 3.3)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 50)
    cajas = [(1, 31, "Monthly panel\n12 series × 48 months (USD)"),
             (53, 31, "Scale-free index: monthly sales ÷\nmean of the previous 12 months"),
             (53, 3, "Pooled seasonal profile (12 values)\nshrunk towards the series' own lag"),
             (1, 3, "Back-transform to USD\nindex × 12-month mean")]
    for x, y, txt in cajas:
        ax.add_patch(FancyBboxPatch((x, y), 46, 14, boxstyle="round,pad=0.4,rounding_size=1.5", fc="#E6F2F4", ec=AZUL, lw=1.2))
        ax.text(x + 23, y + 7, txt, ha="center", va="center", fontsize=9)
    flecha = {"arrowstyle": "-|>", "color": AZUL, "lw": 1.8}
    ax.annotate("", xy=(52, 38), xytext=(48, 38), arrowprops=flecha)
    ax.annotate("", xy=(76, 18), xytext=(76, 30), arrowprops=flecha)
    ax.annotate("", xy=(48, 10), xytext=(52, 10), arrowprops=flecha)
    ax.annotate("", xy=(20, 30), xytext=(20, 18), arrowprops={"arrowstyle": "-|>", "color": AMBAR, "lw": 1.8, "ls": "--"})
    ax.text(22, 24, "recursive step: the forecast of month t\nbecomes history for month t + 1", ha="left", va="center",
            fontsize=10)
    fig.savefig(FIG / "method_diagram.pdf"); plt.close(fig)

    # 4. MAE por bloque mensual y modelo
    ordv = ORDEN[::-1]
    mm = {n: mae_mensual(p, p[n]) / 1000 for n in ORDEN}
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bp = ax.boxplot([mm[n] for n in ordv], vert=False, widths=0.6, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 1.5}, flierprops={"marker": "o", "markersize": 4})
    for patch, n in zip(bp["boxes"], ordv):
        patch.set_facecolor(AMBAR if n == PROPUESTO else CLARO)
    ax.scatter([mm[n].mean() for n in ordv], range(1, len(ordv) + 1), marker="D", color=AZUL, zorder=3, s=28, label="Mean")
    ax.set_yticks(range(1, len(ordv) + 1)); ax.set_yticklabels([EN[n] for n in ordv])
    ax.set_xlabel("Monthly-block MAE, 24 rolling origins (thousand USD)")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(FIG / "mae_boxplot.pdf"); plt.close(fig)

    # 5. Serie pronosticada frente a observada (total de la empresa)
    pm, rec = c["por_mes"], c["rec17"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(pm.Mes, pm.real / 1000, color="black", marker="o", ms=4, label="Observed")
    ax.plot(pm.Mes, pm.pred / 1000, color=AZUL, ls="--", marker="s", ms=4, label="Proposed, one step ahead")
    ax.plot(rec.Mes, rec.rec / 1000, color=AMBAR, ls="-.", marker="^", ms=4, label="Proposed, 12-month recursive (2017)")
    ax.plot(pm.Mes, pm.ref / 1000, color=GRIS, ls=":", marker="x", ms=4, label="Seasonal naive")
    ax.set_xticks(pm.Mes[::3]); ax.set_xticklabels([f"{m:%b}\n{m:%Y}" for m in pm.Mes[::3]], fontsize=10)
    ax.set_xlabel("Month"); ax.set_ylabel("Total sales (thousand USD)")
    ax.set_ylim(0, 150); ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    fig.savefig(FIG / "forecast_total.pdf"); plt.close(fig)

    # 6. Intervalos de la diferencia de MAE frente a cada referencia
    s = c["sig"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    y = np.arange(len(s))
    ax.errorbar(s.dif_MAE, y, xerr=[s.dif_MAE - s.ic_lo, s.ic_hi - s.dif_MAE], fmt="o", color=AZUL, capsize=4, lw=2)
    ax.axvline(0, color="black", lw=1)
    rope = c["N"]["protocolo_b"]["rope_usd"]
    ax.axvspan(-rope, rope, color=AMBAR, alpha=0.3, lw=0, label="ROPE (±1% of mean sales)")
    ax.set_yticks(y); ax.set_yticklabels([EN[n] for n in s.index])
    ax.set_xlabel("MAE difference, proposed − baseline (USD)\nnegative values favour the proposed model")
    ax.set_xlim(-700, 400)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    fig.savefig(FIG / "mae_differences.pdf"); plt.close(fig)

    # 7. Predicho frente a observado y residuos
    real, pr = p.Sales.to_numpy() / 1000, p[PROPUESTO].to_numpy() / 1000
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.2))
    ax[0].scatter(real, pr, s=12, color=AZUL, alpha=0.55, lw=0)
    lim = max(real.max(), pr.max()) * 1.05
    ax[0].plot([0, lim], [0, lim], color=GRIS, lw=1, ls="--")
    ax[0].set_xlim(0, lim); ax[0].set_ylim(0, lim)
    ax[0].set_xlabel("Observed (thousand USD)"); ax[0].set_ylabel("Predicted (thousand USD)")
    ax[1].hist(real - pr, bins=24, color=AZUL, alpha=0.85)
    ax[1].axvline(0, color="black", lw=1)
    ax[1].set_xlabel("Residual (thousand USD)"); ax[1].set_ylabel("Series-months")
    fig.tight_layout(); fig.savefig(FIG / "pred_vs_obs.pdf"); plt.close(fig)

    # 8. Ablación
    ab, ic = c["ablacion"], c["abl_ic"]
    v = list(ab.index)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    y = np.arange(len(v))
    ax.barh(y, ab.MAE[v] / 1000, color=[AMBAR if n == COMPLETO else CLARO for n in v], edgecolor=AZUL,
            xerr=[[(ab.MAE[n] - ic[n][0]) / 1000 for n in v], [(ic[n][1] - ab.MAE[n]) / 1000 for n in v]], capsize=3)
    ax.set_yticks(y); ax.set_yticklabels([EN_AB[n] for n in v], fontsize=10.5)
    ax.set_xlabel("Rolling-origin MAE (thousand USD), 95% block-bootstrap CI", fontsize=11)
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "ablation.pdf"); plt.close(fig)

    # 9. Sensibilidad: peso w y ventana de entrenamiento
    sw, sv = c["sens_w"], c["sens_v"]
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.2))
    for e, col, ls, mk, lab in [("media", AZUL, "-", "o", "Mean profile"), ("mediana", AMBAR, "--", "s", "Median profile")]:
        g = sw[sw.estadistico == e]
        ax[0].plot(g.w, g.MAE / 1000, color=col, ls=ls, marker=mk, ms=5, label=lab)
    ax[0].set_xlabel("Weight w of the series' own lag"); ax[0].set_ylabel("Rolling-origin MAE\n(thousand USD)")
    ax[0].legend(frameon=False, fontsize=10)
    ax[1].plot(range(len(sv)), sv.MAE / 1000, color=AZUL, marker="o")
    ax[1].axhline(c["principal"].MAE[REFERENCIA] / 1000, color=GRIS, ls=":", label="Seasonal naive")
    ax[1].set_xticks(range(len(sv))); ax[1].set_xticklabels(["12", "18", "24", "all"])
    ax[1].set_xlabel("Training window\n(months)"); ax[1].set_ylabel("Rolling-origin MAE\n(thousand USD)")
    ax[1].legend(frameon=False, fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "sensitivity.pdf"); plt.close(fig)

    # 10. Interpretación: el perfil estacional (los 12 parámetros) y el error por serie
    perfil = np.r_[c["perfil"][1:], c["perfil"][0]]      # enero ... diciembre
    ps = c["por_serie"]
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.3))
    ax[0].bar(range(12), perfil, color=CLARO, edgecolor=AZUL)
    ax[0].axhline(1, color=GRIS, lw=1, ls="--")
    ax[0].set_xticks(range(12)); ax[0].set_xticklabels(INICIALES, fontsize=10)
    ax[0].set_xlabel("Month of the year"); ax[0].set_ylabel("Seasonal profile\n(sales index)")
    ax[0].grid(axis="x", visible=False)
    ax[1].scatter(ps.venta_media / 1000, ps.WAPE, color=AZUL, s=40, label="Proposed", zorder=3)
    ax[1].scatter(ps.venta_media / 1000, ps.WAPE_ref, color=GRIS, marker="x", s=40, label="Seasonal naive", zorder=3)
    ax[1].set_xlabel("Mean monthly sales\n(thousand USD)", fontsize=11); ax[1].set_ylabel("WAPE per series (%)")
    ax[1].set_ylim(0, None); ax[1].legend(frameon=False, fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "profile_errors.pdf"); plt.close(fig)
    print(f"Figuras en {FIG}")


if __name__ == "__main__":
    if "--figuras" in sys.argv:
        with open(RES / "cache.pkl", "rb") as f:
            cache = pickle.load(f)
    else:
        cache = calcular()
    figuras(cache)
