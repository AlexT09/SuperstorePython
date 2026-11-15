"""
experiments/ejecutar.py — Validación anidada de todos los modelos en todos los conjuntos de datos.

Bucle externo (estima el desempeño): origen móvil. En cada uno de los últimos 24 meses se entrena con
todos los meses anteriores y se pronostica ese mes para todas las series.
Bucle interno (elige hiperparámetros): dentro de cada conjunto de entrenamiento, validación cruzada
temporal con pliegues de 3 meses (hasta 4), con el método de optimización que corresponda.

Sample Superstore: los 7 modelos base y el modelo original con los 4 métodos de optimización
(28 combinaciones de referencia + 4 del original). Conjuntos adicionales: búsqueda bayesiana.
Además, sin ajuste: ingenuo estacional y Theta; con búsqueda bayesiana: HistGradientBoosting y las
variantes de ablación del modelo original.

Salidas en experiments/resultados/:
    pred_<conjunto>.csv     una fila por serie y mes evaluado, una columna por modelo|optimizador (unidades originales)
    tabla_maestra.csv       una fila por conjunto, origen, modelo y optimizador: hiperparámetros elegidos,
                            puntaje interno, evaluaciones, segundos, historial de la búsqueda y semilla

Ejecutar desde la raíz:  python experiments/ejecutar.py [conjunto ...]      (≈ 25 min en 8 núcleos)
"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

import config as C
from model.features import FEATURES, TARGET, a_ventas, agregar_variables, preprocesador
from src.optimizacion import optimizar
from src.original_model import PerfilEstacionalJerarquico

LAMS_FIJOS = C.REJILLA_ORIGINAL["lam"]


def cargar(conjunto):
    """Devuelve (panel, tabla con variables sin filas incompletas, meses de evaluación)."""
    panel = pd.read_csv(C.CONJUNTOS[conjunto], parse_dates=["Mes"])
    panel["Category"], panel["Region"] = panel.Category.astype(str), panel.Region.astype(str)
    datos = agregar_variables(panel).dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    origenes = np.sort(datos.Mes.unique())[-C.N_ORIGENES:]
    return panel, datos, origenes


def cv_interna(tr):
    """Pliegues de 3 meses (hasta 4), dejando al menos 6 meses para entrenar en el primero."""
    n_meses, n_series = tr.Mes.nunique(), tr.groupby(["Category", "Region"]).ngroups
    return TimeSeriesSplit(n_splits=min(4, (n_meses - 6) // 3), test_size=3 * n_series)


def con_pipeline(est):
    """Preprocesamiento (one-hot + estandarización) dentro del Pipeline: se ajusta en cada pliegue."""
    return Pipeline([("pre", preprocesador()), ("modelo", est)])


def prefijo(d):
    return {f"modelo__{k}": v for k, v in d.items()}


def theta(panel, origen):
    """Método Theta por serie (statsmodels), con toda la historia previa al origen."""
    from statsmodels.tsa.forecasting.theta import ThetaModel
    out = {}
    for clave, g in panel[panel.Mes < origen].groupby(["Category", "Region"]):
        y = g.sort_values("Mes").Sales.to_numpy(float)
        try:
            out[clave] = float(ThetaModel(y, period=12, deseasonalize=len(y) >= 24).fit().forecast(1).iloc[0])
        except Exception:
            out[clave] = float(y[-12]) if len(y) >= 12 else float(y[-1])
    return out


def un_origen(conjunto, panel, datos, origen):
    """Todo el trabajo de un mes de evaluación. Devuelve (predicciones, filas de la tabla maestra)."""
    warnings.filterwarnings("ignore")
    tr, te = datos[datos.Mes < origen], datos[datos.Mes == origen]
    cv = cv_interna(tr)
    pred = te[["Category", "Region", "Mes", "Sales", "prom_12m"]].copy()
    maestra = []

    def registrar(modelo, metodo, r, est_final, X_tr, X_te):
        t0 = time.perf_counter()
        est_final.fit(X_tr, tr[TARGET])
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        p = est_final.predict(X_te)
        t_inf = time.perf_counter() - t0
        pred[f"{modelo}|{metodo}"] = a_ventas(p, te)
        maestra.append({"conjunto": conjunto, "origen": pd.Timestamp(origen), "modelo": modelo, "optimizador": metodo,
                        "hiperparametros": json.dumps({k.replace("modelo__", ""): v for k, v in r["params"].items()}, default=str),
                        "puntaje_interno": r["puntaje"], "evaluaciones": len(r["historial"]),
                        "busqueda_s": r["segundos"], "ajuste_ms": t_fit * 1e3, "inferencia_ms": t_inf * 1e3,
                        "historial": json.dumps([round(v, 6) for v in r["historial"]]),
                        "diversidad": json.dumps(r.get("diversidad", [])), "semilla": C.SEMILLA})

    # referencias sin ajuste
    pred[f"{C.INGENUO}|-"] = te.lag_12.to_numpy()
    th = theta(panel, origen)
    pred[f"{C.THETA}|-"] = [max(0.0, th[(c, r)]) for c, r in zip(te.Category, te.Region)]

    for metodo in C.OPTIMIZADORES[conjunto]:
        for nombre, (est, rejilla, espacio) in C.MODELOS.items():
            pipe = con_pipeline(clone(est))
            r = optimizar(metodo, pipe, prefijo(rejilla), prefijo(espacio), tr[FEATURES], tr[TARGET], cv,
                          n=C.PRESUPUESTO, semilla=C.SEMILLA)
            registrar(nombre, metodo, r, clone(pipe).set_params(**r["params"]), tr[FEATURES], te[FEATURES])
        r = optimizar(metodo, PerfilEstacionalJerarquico(), C.REJILLA_ORIGINAL, C.ESPACIO_ORIGINAL, tr, tr[TARGET], cv,
                      n=C.PRESUPUESTO, semilla=C.SEMILLA)
        registrar(C.ORIGINAL, metodo, r, PerfilEstacionalJerarquico(**r["params"]), tr, te)

    # método adicional reciente y ablación: búsqueda bayesiana con el mismo presupuesto
    pipe = con_pipeline(clone(C.EST_HGB))
    r = optimizar("bayes", pipe, None, prefijo(C.ESPACIO_HGB), tr[FEATURES], tr[TARGET], cv, n=C.PRESUPUESTO, semilla=C.SEMILLA)
    registrar(C.HGB, "bayes", r, clone(pipe).set_params(**r["params"]), tr[FEATURES], te[FEATURES])
    for nombre, (est, ajusta) in C.ABLACION.items():
        if ajusta:
            r = optimizar("bayes", est, None, C.ESPACIO_ORIGINAL, tr, tr[TARGET], cv, n=C.PRESUPUESTO, semilla=C.SEMILLA)
            est = clone(est).set_params(**r["params"])
        pred[f"abl|{nombre}"] = a_ventas(clone(est).fit(tr, tr[TARGET]).predict(te), te)

    # sensibilidad: lam fijo, sin búsqueda
    for lam in LAMS_FIJOS:
        pred[f"lam|{lam:.6g}"] = a_ventas(PerfilEstacionalJerarquico(lam=lam).fit(tr, tr[TARGET]).predict(te), te)
    return pred, maestra


def main(conjuntos):
    C.RES.mkdir(parents=True, exist_ok=True)
    for conjunto in conjuntos:
        t0 = time.perf_counter()
        panel, datos, origenes = cargar(conjunto)
        salida = Parallel(n_jobs=-1)(delayed(un_origen)(conjunto, panel, datos, o) for o in origenes)
        pd.concat([p for p, _ in salida], ignore_index=True).round(4).to_csv(C.RES / f"pred_{conjunto}.csv", index=False)
        nueva = pd.DataFrame([f for _, m in salida for f in m])
        ruta = C.RES / "tabla_maestra.csv"
        if ruta.exists():
            previa = pd.read_csv(ruta, parse_dates=["origen"])
            nueva = pd.concat([previa[previa.conjunto != conjunto], nueva], ignore_index=True)
        nueva.to_csv(ruta, index=False)
        print(f"{conjunto}: {len(origenes)} orígenes en {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(C.CONJUNTOS))
