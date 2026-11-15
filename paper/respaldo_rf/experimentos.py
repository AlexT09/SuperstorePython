"""
paper/experimentos.py — Reproduce todas las tablas, pruebas estadísticas y figuras del artículo.

Usa los mismos datos, variables, rejillas y particiones que model/train_model.py (los importa),
y añade lo que exige el artículo: regresión bayesiana como baseline adicional, incertidumbre por
bloque mensual, pruebas de significancia (Diebold–Mariano–HLN, Giacomini–White, t bayesiana
correlacionada con ROPE) con corrección de Holm, ablación, sensibilidad, costo e interpretabilidad.

Ejecutar desde la raíz del repositorio:
    python paper/experimentos.py            # calcula todo (~2 min) y genera las figuras
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model.features import CATEGORICAS, FEATURES, NUMERICAS, SEED, TARGET, agregar_variables  # noqa: E402
from model.train_model import ANIO_PRUEBA, CV, MODELOS, REFERENCIA, con_prefijo, sin_prefijo  # noqa: E402

PAPER = Path(__file__).resolve().parent
RES, FIG = PAPER / "resultados", PAPER / "figures"
PROPUESTO = "Random Forest"          # elegido por validación cruzada en el proyecto (no por la prueba)
BAYES = "Regresión bayesiana"
ALPHA = 0.05
ROPE_PCT = 0.01                      # región de equivalencia práctica: ±1 % de la venta media de entrenamiento
N_BOOT = 10000
BLOQUE = 3                           # longitud del bloque (meses) del bootstrap
SEMILLAS = list(range(10))

# Nombres en inglés para las figuras
EN = {REFERENCIA: "Seasonal naive", "Ridge": "Ridge", "Lasso": "Lasso", BAYES: "Bayesian ridge", "KNN": "KNN",
      "Árbol de decisión": "Decision tree", "Random Forest": "Random forest (proposed)", "XGBoost": "XGBoost",
      "SVR": "SVR"}


# ----------------------------------------------------------------------------- utilidades
def pre(cats, nums):
    partes = []
    if cats:
        partes.append(("cat", OneHotEncoder(handle_unknown="ignore"), cats))
    partes.append(("num", StandardScaler(), nums))
    return ColumnTransformer(partes)


def buscar(est, grid, X, y, cats=CATEGORICAS, nums=NUMERICAS, cv=CV):
    pipe = Pipeline([("pre", pre(cats, nums)), ("modelo", est)])
    s = GridSearchCV(pipe, con_prefijo(grid), cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    t0 = time.perf_counter()
    s.fit(X[cats + nums], y)
    return s, time.perf_counter() - t0


def metricas(real, pred, escala):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    e = real - pred
    return {"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()), "WAPE": np.abs(e).sum() / real.sum() * 100,
            "MASE": np.abs(e).mean() / escala, "R2": r2_score(real, pred)}


def desvio_total(filas, pred):
    return (np.sum(pred) / filas.Sales.sum() - 1) * 100


def mae_mensual(filas, pred):
    """MAE de cada mes (promedio sobre las 12 series): los 12 bloques temporales de la prueba."""
    return (filas.assign(e=np.abs(filas.Sales.to_numpy() - pred)).groupby("Mes").e.mean()).to_numpy()


def recursivo(predecir, panel_hist, meses):
    """Proyección recursiva genérica: cada mes proyectado pasa a ser historia del siguiente."""
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
    omega = Z.T @ Z / n
    stat = float(n * zbar @ np.linalg.pinv(omega) @ zbar)
    return stat, stats.chi2.sf(stat, df=H.shape[1])


def bayes_rope(x, rope, rho):
    """t correlacionada bayesiana (Corani y Benavoli): x = puntaje baseline − propuesto (positivo = mejora).

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
    medias = d[idx.reshape(N_BOOT, -1)[:, :T]].mean(axis=1)
    return np.percentile(medias, [2.5, 97.5])


def contrastar(nombre_base, err_prop, err_base, filas, rope, rng):
    """Todas las pruebas del modelo propuesto frente a un baseline (o del modelo completo frente a una variante)."""
    f = filas.assign(d=np.abs(err_prop) - np.abs(err_base), d2=err_prop ** 2 - err_base ** 2)
    d = f.groupby("Mes").d.mean().to_numpy()     # diferencial de pérdida absoluta por mes; negativo = propuesto mejor
    d2 = f.groupby("Mes").d2.mean().to_numpy()
    dm, p_dm = dm_hln(d)
    dm2, p_dm2 = dm_hln(d2)
    gw, p_gw = giacomini_white(d)
    mejor, igual, peor = bayes_rope(-d, rope, rho=1 / d.size)
    lo, hi = ic_bootstrap(d, rng)
    w = stats.wilcoxon(d)
    return {"baseline": nombre_base, "dif_MAE": d.mean(), "ic_lo": lo, "ic_hi": hi,
            "d_cohen": d.mean() / d.std(ddof=1), "DM": dm, "p_DM": p_dm, "DM_sq": dm2, "p_DM_sq": p_dm2,
            "GW": gw, "p_GW": p_gw, "P_mejor": mejor, "P_rope": igual, "P_peor": peor,
            "p_wilcoxon": w.pvalue, "shapiro_p": stats.shapiro(d).pvalue}


def holm(df, cols):
    for c in cols:
        df[c + "_holm"] = multipletests(df[c], alpha=ALPHA, method="holm")[1]
    return df


# ----------------------------------------------------------------------------- cálculo
def calcular():
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(SEED)
    N = {}  # números sueltos que cita el texto

    # --- Datos y descripción (EDA) --------------------------------------------------------------
    crudo = pd.read_csv(ROOT / "data" / "raw" / "Sample - Superstore.csv", encoding="latin1")
    lineas = pd.read_csv(ROOT / "data" / "superstore_transformado.csv", parse_dates=["Order_Date"])
    panel = pd.read_csv(ROOT / "data" / "ventas_mensuales.csv", parse_dates=["Mes"])
    datos = agregar_variables(panel).dropna(subset=FEATURES + [TARGET])
    train = datos[datos.Mes.dt.year < ANIO_PRUEBA]
    test = datos[datos.Mes.dt.year == ANIO_PRUEBA]

    def desc(s):
        q1, q3 = s.quantile([.25, .75])
        return {"n": int(s.size), "media": s.mean(), "mediana": s.median(), "sd": s.std(), "min": s.min(),
                "max": s.max(), "asimetria": s.skew(), "faltantes": int(s.isna().sum()),
                "atipicos_iqr": int(((s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))).sum())}

    anual = panel.groupby(panel.Mes.dt.year).Sales.sum()
    por_mes = panel.groupby(panel.Mes.dt.month).Sales.sum()
    N["eda"] = {
        "filas_crudo": len(crudo), "columnas_crudo": crudo.shape[1], "duplicados": int(crudo.duplicated().sum()),
        "faltantes_crudo": int(crudo.isna().sum().sum()),
        "fecha_min": str(lineas.Order_Date.min().date()), "fecha_max": str(lineas.Order_Date.max().date()),
        "linea": {c: desc(lineas[c]) for c in ["Sales", "Quantity", "Discount", "Profit"]},
        "panel_sales": desc(panel.Sales), "indice": desc(datos[TARGET]),
        "panel_ceros": int((panel.Sales == 0).sum()),
        "anual": anual.round(0).to_dict(), "crec_anual_pct": (anual.pct_change() * 100).round(1).dropna().to_dict(),
        "cuota_categoria": (panel.groupby("Category").Sales.sum() / panel.Sales.sum() * 100).round(1).to_dict(),
        "cuota_region": (panel.groupby("Region").Sales.sum() / panel.Sales.sum() * 100).round(1).to_dict(),
        "cuota_mes": (por_mes / por_mes.sum() * 100).round(1).to_dict(),
        "cuota_sep_dic": float(por_mes.loc[[9, 11, 12]].sum() / por_mes.sum() * 100),
        "media_serie": panel.groupby(["Category", "Region"]).Sales.mean().round(0)
                            .rename(lambda k: k, level=0).reset_index().values.tolist(),
        "corr_linea": lineas[["Sales", "Quantity", "Discount", "Profit"]].corr(method="spearman").Sales.round(3).to_dict(),
        "corr_indice": datos[[TARGET] + NUMERICAS].corr(method="spearman")[TARGET].round(3).to_dict(),
        "corr_dolares": datos[["Sales", "lag_12", "lag_1", "prom_3m", "prom_12m"]].corr(method="spearman").Sales.round(3).to_dict(),
        "cv_series_dolares": float(panel.groupby(["Category", "Region"]).Sales.mean().std()
                                   / panel.groupby(["Category", "Region"]).Sales.mean().mean()),
    }
    escala = float(np.abs(train.Sales - train.lag_12).mean())      # MAE del ingenuo estacional en entrenamiento (MASE)
    rope = ROPE_PCT * float(train.Sales.mean())
    N["protocolo"] = {"n_train": len(train), "n_test": len(test), "escala_mase": escala, "rope_usd": rope,
                      "venta_media_train": float(train.Sales.mean()), "venta_media_test": float(test.Sales.mean())}

    a_usd = lambda ind, f: np.clip(np.asarray(ind) * f.prom_12m.to_numpy(), 0, None)  # noqa: E731
    meses_prueba = pd.date_range(f"{ANIO_PRUEBA}-01-01", periods=12, freq="MS")
    hist_2016 = panel[panel.Mes.dt.year < ANIO_PRUEBA]
    real = test.Sales.to_numpy()

    # --- Modelos: los 7 del proyecto + regresión bayesiana, misma partición, mismo pipeline --------
    modelos = dict(MODELOS)
    modelos[BAYES] = (BayesianRidge(), {"alpha_1": [1e-6, 1e-3, 1.0], "lambda_1": [1e-6, 1e-3, 1.0]})
    orden = [REFERENCIA, "Ridge", "Lasso", BAYES, "KNN", "Árbol de decisión", "XGBoost", "SVR", PROPUESTO]

    pred, pred_rec, busc, filas = {REFERENCIA: test.lag_12.to_numpy()}, {}, {}, []
    # ingenuo estacional a 12 meses = mismo mes de 2016 (coincide con el de un paso)
    pred_rec[REFERENCIA] = pred[REFERENCIA]
    for nombre, (est, grid) in modelos.items():
        s, seg = buscar(est, grid, train, train[TARGET])
        busc[nombre] = s
        pred[nombre] = a_usd(s.predict(test[FEATURES]), test)
        pred_rec[nombre] = recursivo(lambda f, s=s: a_usd(s.predict(f[FEATURES]), f), hist_2016, meses_prueba)
        # costo: reajuste de la configuración elegida y predicción de las 144 filas, en un solo hilo
        t_fit = []
        for _ in range(5):
            m = clone(s.best_estimator_)
            t0 = time.perf_counter(); m.fit(train[FEATURES], train[TARGET]); t_fit.append(time.perf_counter() - t0)
        t_inf = []
        for _ in range(30):
            t0 = time.perf_counter(); s.best_estimator_.predict(test[FEATURES]); t_inf.append(time.perf_counter() - t0)
        filas.append({"modelo": nombre, "cv_MAE_indice": -s.best_score_, "evaluaciones": len(s.cv_results_["params"]),
                      "busqueda_s": seg, "ajuste_ms": np.median(t_fit) * 1e3, "inferencia_ms": np.median(t_inf) * 1e3,
                      "tamano_kB": len(pickle.dumps(s.best_estimator_)) / 1024,
                      "hiperparametros": json.dumps(sin_prefijo(s.best_params_), ensure_ascii=False),
                      "rejilla": json.dumps(grid, ensure_ascii=False)})
        print(f"  {nombre:20s} CV {-s.best_score_:.4f} · {seg:5.1f}s")
    costo = pd.DataFrame(filas).set_index("modelo")
    costo.loc[REFERENCIA] = {"evaluaciones": 0, "busqueda_s": 0, "ajuste_ms": 0, "inferencia_ms": 0, "tamano_kB": 0}

    mm = {n: mae_mensual(test, pred[n]) for n in orden}
    principal = pd.DataFrame([{"modelo": n, **metricas(real, pred[n], escala), "MAE_sd": mm[n].std(ddof=1),
                               "desvio_total_pct": desvio_total(test, pred[n]),
                               "rec_MAE": np.abs(real - pred_rec[n]).mean(),
                               "rec_desvio_total_pct": desvio_total(test, pred_rec[n]),
                               "cv_MAE_indice": costo.cv_MAE_indice.get(n, np.nan)} for n in orden]).set_index("modelo")
    # mejora del modelo propuesto en entrenamiento vs. prueba (sobreajuste)
    p_tr = a_usd(busc[PROPUESTO].predict(train[FEATURES]), train)
    N["sobreajuste"] = {"MAE_train": float(np.abs(train.Sales - p_tr).mean()), "MAE_test": float(principal.MAE[PROPUESTO])}

    # --- Significancia: propuesto vs. cada baseline --------------------------------------------------
    err = {n: real - pred[n] for n in orden}
    baselines = [n for n in orden if n != PROPUESTO]
    sig = holm(pd.DataFrame([contrastar(b, err[PROPUESTO], err[b], test, rope, rng) for b in baselines]),
               ["p_DM", "p_DM_sq", "p_GW", "p_wilcoxon"]).set_index("baseline")
    # mismas pruebas con la proyección recursiva a 12 meses (h hasta 12: se reporta solo el diferencial y su IC)
    err_rec = {n: real - pred_rec[n] for n in orden}
    sig_rec = pd.DataFrame([contrastar(b, err_rec[PROPUESTO], err_rec[b], test, rope, rng) for b in baselines]
                           ).set_index("baseline")[["dif_MAE", "ic_lo", "ic_hi", "P_mejor", "P_rope", "P_peor"]]

    # --- Semillas (modelos con componente aleatorio) -------------------------------------------------
    sem = {}
    for n in ["Árbol de decisión", PROPUESTO, "XGBoost"]:
        v = []
        for sd in SEMILLAS:
            m = clone(busc[n].best_estimator_).set_params(modelo__random_state=sd).fit(train[FEATURES], train[TARGET])
            v.append(np.abs(real - a_usd(m.predict(test[FEATURES]), test)).mean())
        sem[n] = v
    semillas = pd.DataFrame(sem)

    # --- Ablación: se quita o reemplaza un componente del modelo completo ---------------------------
    rf, grid_rf = MODELOS[PROPUESTO]
    variantes = {
        "Sin estacionalidad (seno/coseno del mes)": (CATEGORICAS, ["r_lag_12", "r_lag_1", "r_prom_3m"]),
        "Sin rezago estacional (mes del año anterior)": (CATEGORICAS, ["r_lag_1", "r_prom_3m", "mes_sin", "mes_cos"]),
        "Sin historia reciente (mes anterior, prom. 3 meses)": (CATEGORICAS, ["r_lag_12", "mes_sin", "mes_cos"]),
        "Sin identificadores de serie (categoría, región)": ([], NUMERICAS),
    }
    pred_ab, rec_ab, cv_ab = {"Modelo completo": pred[PROPUESTO]}, {"Modelo completo": pred_rec[PROPUESTO]}, {}
    for nombre, (cats, nums) in variantes.items():
        s, _ = buscar(clone(rf), grid_rf, train, train[TARGET], cats, nums)
        cols = cats + nums
        pred_ab[nombre] = a_usd(s.predict(test[cols]), test)
        rec_ab[nombre] = recursivo(lambda f, s=s, cols=cols: a_usd(s.predict(f[cols]), f), hist_2016, meses_prueba)
        cv_ab[nombre] = -s.best_score_
        print(f"  ablación: {nombre}")
    # sin índice: objetivo y rezagos en dólares
    nums_usd = ["lag_12", "lag_1", "prom_3m", "mes_sin", "mes_cos"]
    s, _ = buscar(clone(rf), grid_rf, train, train.Sales, CATEGORICAS, nums_usd)
    cols = CATEGORICAS + nums_usd
    n_usd = "Sin índice relativo (objetivo y rezagos en dólares)"
    pred_ab[n_usd] = np.clip(s.predict(test[cols]), 0, None)
    rec_ab[n_usd] = recursivo(lambda f, s=s, cols=cols: np.clip(s.predict(f[cols]), 0, None), hist_2016, meses_prueba)
    # sin modelo global: un bosque por serie (24 filas de entrenamiento cada uno)
    n_loc = "Sin modelo global (un modelo por serie)"
    cv_local = TimeSeriesSplit(n_splits=4, test_size=3)
    locales = {}
    for clave, g in train.groupby(["Category", "Region"]):
        locales[clave], _ = buscar(clone(rf), grid_rf, g, g[TARGET], [], NUMERICAS, cv_local)

    def pred_local(f):
        out = np.empty(len(f))
        for clave, idx in f.groupby(["Category", "Region"]).indices.items():
            g = f.iloc[idx]
            out[idx] = a_usd(locales[clave].predict(g[NUMERICAS]), g)
        return out
    pred_ab[n_loc] = pred_local(test)
    rec_ab[n_loc] = recursivo(pred_local, hist_2016, meses_prueba)
    print("  ablación: índice y modelo global")

    e_full = real - pred_ab["Modelo completo"]
    abl_sig = holm(pd.DataFrame([contrastar(v, e_full, real - pred_ab[v], test, rope, rng)
                                 for v in pred_ab if v != "Modelo completo"]), ["p_DM", "p_GW"]).set_index("baseline")
    ablacion = pd.DataFrame([{"variante": v, **metricas(real, pred_ab[v], escala),
                              "MAE_sd": mae_mensual(test, pred_ab[v]).std(ddof=1),
                              "desvio_total_pct": desvio_total(test, pred_ab[v]),
                              "rec_MAE": np.abs(real - rec_ab[v]).mean(),
                              "rec_desvio_total_pct": desvio_total(test, rec_ab[v])} for v in pred_ab]).set_index("variante")
    ablacion = ablacion.join(abl_sig[["dif_MAE", "ic_lo", "ic_hi", "p_DM_holm", "p_GW_holm", "P_mejor", "P_rope", "P_peor"]])
    abl_ic = {v: ic_bootstrap(mae_mensual(test, pred_ab[v]), rng) for v in pred_ab}

    # --- Análisis exploratorio (posterior al protocolo): error y sesgo del total mensual de la empresa ----
    tot = lambda p: test.assign(p=p).groupby("Mes").p.sum().to_numpy()  # noqa: E731
    real_t, prop_t = tot(real), tot(pred[PROPUESTO])
    expl = []
    for nombre, p in [(PROPUESTO, pred[PROPUESTO]), (REFERENCIA, pred[REFERENCIA]), ("SVR", pred["SVR"]),
                      (n_usd, pred_ab[n_usd])]:
        sesgo = tot(p) - real_t
        lo_s, hi_s = ic_bootstrap(sesgo, rng)
        fila = {"modelo": nombre, "MAE_total": np.abs(sesgo).mean(), "sesgo_medio": sesgo.mean(),
                "sesgo_lo": lo_s, "sesgo_hi": hi_s, "desvio_total_pct": desvio_total(test, p)}
        if nombre != PROPUESTO:
            d = np.abs(real_t - prop_t) - np.abs(sesgo)
            lo, hi = ic_bootstrap(d, rng)
            fila.update({"dif_MAE_total": d.mean(), "ic_lo": lo, "ic_hi": hi, "p_DM": dm_hln(d)[1]})
        expl.append(fila)
    exploratorio = pd.DataFrame(expl).set_index("modelo")

    # --- Sensibilidad -------------------------------------------------------------------------------
    cvr = pd.DataFrame(busc[PROPUESTO].cv_results_)
    rejilla_rf = cvr.assign(max_depth=cvr["param_modelo__max_depth"].map(lambda v: "None" if v is None else str(v)),
                            min_samples_leaf=cvr["param_modelo__min_samples_leaf"].astype(int),
                            max_features=cvr["param_modelo__max_features"].astype(float),
                            cv_MAE=-cvr.mean_test_score, cv_sd=cvr.std_test_score)[
        ["max_depth", "min_samples_leaf", "max_features", "cv_MAE", "cv_sd"]]
    # error de prueba de cada configuración de la rejilla (solo para describir la sensibilidad; no interviene en la elección)
    te = []
    for params in busc[PROPUESTO].cv_results_["params"]:
        m = clone(busc[PROPUESTO].best_estimator_).set_params(**params).fit(train[FEATURES], train[TARGET])
        te.append(np.abs(real - a_usd(m.predict(test[FEATURES]), test)).mean())
    rejilla_rf["test_MAE"] = te
    # curva de aprendizaje: últimos k meses de entrenamiento, hiperparámetros fijos
    curva = []
    for k in [6, 9, 12, 15, 18, 21, 24]:
        ini = pd.Timestamp(f"{ANIO_PRUEBA}-01-01") - pd.DateOffset(months=k)
        tr = train[train.Mes >= ini]
        m = clone(busc[PROPUESTO].best_estimator_).fit(tr[FEATURES], tr[TARGET])
        p = a_usd(m.predict(test[FEATURES]), test)
        curva.append({"meses": k, "filas": len(tr), "MAE": np.abs(real - p).mean(),
                      "MAE_sd": mae_mensual(test, p).std(ddof=1)})
    curva = pd.DataFrame(curva)

    # --- Interpretabilidad y errores -----------------------------------------------------------------
    pi = permutation_importance(busc[PROPUESTO].best_estimator_, test[FEATURES], test[TARGET], n_repeats=30,
                                random_state=SEED, scoring="neg_mean_absolute_error")
    importancia = pd.DataFrame({"variable": FEATURES, "importancia": pi.importances_mean, "sd": pi.importances_std}
                               ).sort_values("importancia", ascending=False)
    t = test.assign(pred=pred[PROPUESTO], ref=pred[REFERENCIA])
    t["ae"], t["ae_ref"], t["e"] = (t.Sales - t.pred).abs(), (t.Sales - t.ref).abs(), t.Sales - t.pred
    por_serie = t.groupby(["Category", "Region"]).agg(venta_media=("Sales", "mean"), MAE=("ae", "mean"),
                                                      MAE_ref=("ae_ref", "mean"), sesgo=("e", "mean")).reset_index()
    por_serie["WAPE"] = por_serie.MAE / por_serie.venta_media * 100
    por_mes_err = t.groupby("Mes").agg(real=("Sales", "sum"), pred=("pred", "sum"), ref=("ref", "sum"),
                                       MAE=("ae", "mean"), MAE_ref=("ae_ref", "mean")).reset_index()
    por_mes_err["rec"] = test.assign(r=pred_rec[PROPUESTO]).groupby("Mes").r.sum().to_numpy()
    t["tercil"] = pd.qcut(t.Sales, 3, labels=["bajo", "medio", "alto"])
    por_tercil = t.groupby("tercil", observed=True).agg(venta_media=("Sales", "mean"), MAE=("ae", "mean"),
                                                        sesgo=("e", "mean"), MAE_ref=("ae_ref", "mean")).reset_index()
    peores = t.nlargest(5, "ae")[["Category", "Region", "Mes", "Sales", "pred", "e"]]
    N["errores"] = {"corr_ae_ventas": float(stats.spearmanr(t.Sales, t.ae)[0]),
                    "series_mejor_que_ref": int((por_serie.MAE < por_serie.MAE_ref).sum()),
                    "meses_mejor_que_ref": int((por_mes_err.MAE < por_mes_err.MAE_ref).sum()),
                    "residuo_media": float(t.e.mean()), "residuo_asimetria": float(t.e.skew()),
                    "total_mensual_MAPE": float(((por_mes_err.pred - por_mes_err.real).abs() / por_mes_err.real).mean() * 100),
                    "total_mensual_MAPE_ref": float(((por_mes_err.ref - por_mes_err.real).abs() / por_mes_err.real).mean() * 100),
                    "total_mensual_MAPE_rec": float(((por_mes_err.rec - por_mes_err.real).abs() / por_mes_err.real).mean() * 100)}
    N["arboles"] = {"n": 300, "nodos_medio": float(np.mean([e.tree_.node_count for e in
                                                             busc[PROPUESTO].best_estimator_[-1].estimators_])),
                    "prof_max": int(max(e.tree_.max_depth for e in busc[PROPUESTO].best_estimator_[-1].estimators_))}

    c = dict(N=N, orden=orden, principal=principal, costo=costo, sig=sig, sig_rec=sig_rec, semillas=semillas,
             ablacion=ablacion, abl_ic=abl_ic, rejilla_rf=rejilla_rf, curva=curva, importancia=importancia,
             por_serie=por_serie, por_mes_err=por_mes_err, por_tercil=por_tercil, peores=peores, mm=mm,
             pred=pred, pred_rec=pred_rec, test=test, panel=panel, datos=datos, pred_ab=pred_ab,
             exploratorio=exploratorio)
    RES.mkdir(exist_ok=True)
    with open(RES / "cache.pkl", "wb") as f:
        pickle.dump(c, f)
    for k in ["principal", "costo", "sig", "sig_rec", "semillas", "ablacion", "rejilla_rf", "curva", "importancia",
              "por_serie", "por_mes_err", "por_tercil", "peores", "exploratorio"]:
        c[k].round(4).to_csv(RES / f"{k}.csv")
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
    AZUL, AMBAR, GRIS, ROJO = "#1F3A5F", "#E0A526", "#7A7A7A", "#B03A2E"
    FIG.mkdir(exist_ok=True)
    panel, test, pred, orden = c["panel"], c["test"], c["pred"], c["orden"]
    k = lambda v, _=None: f"{v / 1000:.0f}"  # noqa: E731

    # 1. EDA: serie total y dispersión por categoría
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.4), gridspec_kw={"width_ratios": [1.5, 1]})
    tot = panel.groupby("Mes").Sales.sum()
    ax[0].plot(tot.index, tot.values / 1000, color=AZUL)
    ax[0].axvspan(pd.Timestamp("2017-01-01"), pd.Timestamp("2017-12-31"), color=AMBAR, alpha=0.25, lw=0)
    ax[0].axvspan(pd.Timestamp("2014-01-01"), pd.Timestamp("2014-12-31"), color=GRIS, alpha=0.15, lw=0)
    ax[0].text(pd.Timestamp("2014-07-01"), 112, "history", ha="center", fontsize=10)
    ax[0].text(pd.Timestamp("2016-01-01"), 112, "training", ha="center", fontsize=10)
    ax[0].text(pd.Timestamp("2017-07-01"), 112, "test", ha="center", fontsize=10)
    ax[0].set_ylim(0, 125)
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
    d = c["datos"]
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.2))
    g = [d[d.Mes.dt.month == m].indice for m in range(1, 13)]
    ax[0].boxplot(g, widths=0.6, medianprops={"color": AMBAR, "linewidth": 2}, flierprops={"marker": "o", "markersize": 3})
    ax[0].axhline(1, color=GRIS, lw=1, ls="--")
    ax[0].set_xticks(range(1, 13)); ax[0].set_xticklabels(list("JFMAMJJASOND"), fontsize=10)
    ax[0].set_xlabel("Month of the year"); ax[0].set_ylabel("Sales index (ratio)")
    ax[1].scatter(d.r_lag_12, d.indice, s=12, color=AZUL, alpha=0.5, lw=0)
    ax[1].plot([0, 4], [0, 4], color=GRIS, lw=1, ls="--")
    ax[1].set_xlim(0, 4.2); ax[1].set_ylim(0, 4.2)
    ax[1].set_xlabel("Same month of previous\nyear (ratio)"); ax[1].set_ylabel("Sales index (ratio)")
    fig.tight_layout(); fig.savefig(FIG / "eda_index.pdf"); plt.close(fig)

    # 3. Diagrama del método (ciclo: panel -> variables -> modelo -> ventas -> panel)
    fig, ax = plt.subplots(figsize=(6.5, 3.3)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 50)
    cajas = [(1, 31, "Monthly panel\n12 series × 48 months (USD)"),
             (53, 31, "Scale-free features: lags ÷ 12-month\nmean, month of year (sine, cosine)"),
             (53, 3, "Global random forest (pooled series)\ntarget: sales index"),
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
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ordv = orden[::-1]
    bp = ax.boxplot([c["mm"][n] / 1000 for n in ordv], vert=False, widths=0.6, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 1.5}, flierprops={"marker": "o", "markersize": 4})
    for patch, n in zip(bp["boxes"], ordv):
        patch.set_facecolor(AMBAR if n == PROPUESTO else "#C9D6E3")
    ax.scatter([c["mm"][n].mean() / 1000 for n in ordv], range(1, len(ordv) + 1), marker="D", color=AZUL, zorder=3, s=28,
               label="Mean")
    ax.set_yticks(range(1, len(ordv) + 1)); ax.set_yticklabels([EN[n] for n in ordv])
    ax.set_xlabel("Monthly-block MAE on the 2017 test set (thousand USD)")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(FIG / "mae_boxplot.pdf"); plt.close(fig)

    # 5. Serie pronosticada frente a observada (total de la empresa)
    pm = c["por_mes_err"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(pm.Mes, pm.real / 1000, color="black", marker="o", label="Observed")
    ax.plot(pm.Mes, pm.pred / 1000, color=AZUL, ls="--", marker="s", label="Proposed, one step ahead")
    ax.plot(pm.Mes, pm.rec / 1000, color=AMBAR, ls="-.", marker="^", label="Proposed, 12-month recursive")
    ax.plot(pm.Mes, pm.ref / 1000, color=GRIS, ls=":", marker="x", label="Seasonal naive")
    ax.set_xticks(pm.Mes); ax.set_xticklabels(list("JFMAMJJASOND"))
    ax.set_xlabel("Month of 2017"); ax.set_ylabel("Total sales (thousand USD)")
    ax.set_ylim(0, None); ax.legend(frameon=False, fontsize=10, loc="upper left")
    fig.savefig(FIG / "forecast_total.pdf"); plt.close(fig)

    # 6. Intervalos de la diferencia de MAE frente a cada baseline
    s = c["sig"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    y = np.arange(len(s))
    ax.errorbar(s.dif_MAE, y, xerr=[s.dif_MAE - s.ic_lo, s.ic_hi - s.dif_MAE], fmt="o", color=AZUL, capsize=4, lw=2)
    ax.axvline(0, color="black", lw=1)
    rope = c["N"]["protocolo"]["rope_usd"]
    ax.axvspan(-rope, rope, color=AMBAR, alpha=0.3, lw=0, label="ROPE (±1% of mean sales)")
    ax.set_yticks(y); ax.set_yticklabels([EN[n] for n in s.index])
    ax.set_xlabel("MAE difference, proposed − baseline (USD)\nnegative values favour the proposed model")
    ax.set_xlim(-700, 700)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    fig.savefig(FIG / "mae_differences.pdf"); plt.close(fig)

    # 7. Predicho frente a observado y residuos
    real, p = test.Sales.to_numpy() / 1000, pred[PROPUESTO] / 1000
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.2))
    ax[0].scatter(real, p, s=14, color=AZUL, alpha=0.6, lw=0)
    lim = max(real.max(), p.max()) * 1.05
    ax[0].plot([0, lim], [0, lim], color=GRIS, lw=1, ls="--")
    ax[0].set_xlim(0, lim); ax[0].set_ylim(0, lim)
    ax[0].set_xlabel("Observed (thousand USD)"); ax[0].set_ylabel("Predicted (thousand USD)")
    ax[1].hist(real - p, bins=20, color=AZUL, alpha=0.85)
    ax[1].axvline(0, color="black", lw=1)
    ax[1].set_xlabel("Residual (thousand USD)"); ax[1].set_ylabel("Series-months")
    fig.tight_layout(); fig.savefig(FIG / "pred_vs_obs.pdf"); plt.close(fig)

    # 8. Ablación
    ab, ic = c["ablacion"], c["abl_ic"]
    en_ab = {"Modelo completo": "Full model", "Sin estacionalidad (seno/coseno del mes)": "No month encoding",
             "Sin rezago estacional (mes del año anterior)": "No seasonal lag",
             "Sin historia reciente (mes anterior, prom. 3 meses)": "No recent history",
             "Sin identificadores de serie (categoría, región)": "No series identifiers",
             "Sin índice relativo (objetivo y rezagos en dólares)": "No scale-free index",
             "Sin modelo global (un modelo por serie)": "No pooling (local models)"}
    v = list(ab.index)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    y = np.arange(len(v))
    ax.barh(y, ab.MAE[v] / 1000, color=[AMBAR if n == "Modelo completo" else "#C9D6E3" for n in v], edgecolor=AZUL,
            xerr=[[(ab.MAE[n] - ic[n][0]) / 1000 for n in v], [(ic[n][1] - ab.MAE[n]) / 1000 for n in v]], capsize=3)
    ax.set_yticks(y); ax.set_yticklabels([en_ab[n] for n in v])
    ax.set_xlabel("Test MAE (thousand USD), 95% block-bootstrap CI")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "ablation.pdf"); plt.close(fig)

    # 9. Sensibilidad: rejilla y curva de aprendizaje
    r = c["rejilla_rf"]
    r = r[r.max_features == 0.5].pivot(index="max_depth", columns="min_samples_leaf", values="cv_MAE").loc[["3", "5", "None"]]
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.1))
    im = ax[0].imshow(r.values, cmap="cividis_r", aspect="auto")
    ax[0].set_xticks(range(r.shape[1])); ax[0].set_xticklabels(r.columns)
    ax[0].set_yticks(range(3)); ax[0].set_yticklabels(["3", "5", "no limit"])
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            val = r.values[i, j]
            ax[0].text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=10,
                       color="white" if val > r.values.mean() else "black")
    ax[0].grid(False)
    ax[0].set_xlabel("Minimum samples per leaf", fontsize=12); ax[0].set_ylabel("Maximum depth")
    cu = c["curva"]
    ax[1].errorbar(cu.meses, cu.MAE / 1000, color=AZUL, marker="o")
    ax[1].axhline(c["principal"].MAE[REFERENCIA] / 1000, color=GRIS, ls=":", label="Seasonal naive")
    ax[1].set_xticks(cu.meses)
    ax[1].set_xlabel("Training window (months)", fontsize=12); ax[1].set_ylabel("Test MAE (thousand USD)")
    ax[1].legend(frameon=False, fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "sensitivity.pdf"); plt.close(fig)

    # 10. Importancia de variables y error por serie
    en_var = {"Category": "Category", "Region": "Region", "r_lag_12": "Same month, previous year",
              "r_lag_1": "Previous month", "r_prom_3m": "Mean of last 3 months", "mes_sin": "Month (sine)",
              "mes_cos": "Month (cosine)"}
    imp = c["importancia"].iloc[::-1]
    ps = c["por_serie"].sort_values("venta_media")
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.4))
    ax[0].barh(range(len(imp)), imp.importancia, xerr=imp.sd, color="#C9D6E3", edgecolor=AZUL, capsize=3)
    ax[0].axvline(0, color="black", lw=1)
    ax[0].set_yticks(range(len(imp))); ax[0].set_yticklabels([en_var[v] for v in imp.variable], fontsize=10)
    ax[0].set_xlabel("Permutation importance\n(index MAE increase)", fontsize=11); ax[0].grid(axis="y", visible=False)
    ax[1].scatter(ps.venta_media / 1000, ps.WAPE, color=AZUL, s=40, label="Proposed", zorder=3)
    ax[1].scatter(ps.venta_media / 1000, ps.MAE_ref / ps.venta_media * 100, color=GRIS, marker="x", s=40,
                  label="Seasonal naive", zorder=3)
    ax[1].set_xlabel("Mean monthly sales\n(thousand USD)", fontsize=11); ax[1].set_ylabel("WAPE per series (%)")
    ax[1].set_ylim(0, None); ax[1].legend(frameon=False, fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "importance_errors.pdf"); plt.close(fig)
    print(f"Figuras en {FIG}")


if __name__ == "__main__":
    if "--figuras" in sys.argv:
        with open(RES / "cache.pkl", "rb") as f:
            cache = pickle.load(f)
    else:
        cache = calcular()
    figuras(cache)
