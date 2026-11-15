"""
experiments/artefactos.py — Reentrena y guarda los modelos que usa el dashboard.

Para cada uno de los 7 modelos base toma su mejor combinación (el optimizador con menor MAE del
bucle externo, experiments/resultados/mejores.csv); para el modelo original, la búsqueda bayesiana.
Con ese optimizador se eligen los hiperparámetros sobre toda la historia, se reentrena el pipeline
completo y se guarda con joblib (compress=3). El dashboard solo carga estos archivos: no entrena.

Salidas en artifacts/:
    <modelo>.joblib          pipeline final (preprocesamiento + modelo)
    metadata.json            versiones, semilla, hiperparámetros, esquema de entrada, métricas del bucle
                             externo, radios conformes y resultado de la prueba de ida y vuelta
    proyeccion_2018.csv      proyección recursiva de 2018 de cada modelo
    shapley_global.csv       importancia global (media del valor absoluto de Shapley) de cada modelo
    convergencia.csv         valor del objetivo J del modelo original en cada iteración

Ejecutar desde la raíz (después de analizar.py):  python experiments/artefactos.py
"""
import json
import platform
import unicodedata
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone

import config as C
from ejecutar import cargar, con_pipeline, cv_interna, prefijo
from model.features import FEATURES, TARGET, a_ventas, agregar_variables
from src.explicacion import shapley_exacto
from src.optimizacion import optimizar
from src.original_model import PerfilEstacionalJerarquico

ART = C.ROOT / "artifacts"


def slug(nombre):
    s = unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_").replace("__", "_")


def proyectar(predecir, panel, meses):
    """Proyección recursiva: cada mes proyectado pasa a ser historia del siguiente."""
    hist = panel[["Category", "Region", "Mes", "Sales"]].copy()
    series = hist[["Category", "Region"]].drop_duplicates()
    for mes in meses:
        hist = pd.concat([hist, series.assign(Mes=mes, Sales=np.nan)], ignore_index=True)
        f = agregar_variables(hist)
        f = f[f.Mes == mes]
        hist = pd.concat([hist[hist.Mes != mes], f[["Category", "Region", "Mes"]].assign(Sales=predecir(f))], ignore_index=True)
    return hist[hist.Mes.isin(meses)].rename(columns={"Sales": "Proyeccion"})


def main():
    warnings.filterwarnings("ignore")
    ART.mkdir(exist_ok=True)
    panel, datos, origenes = cargar(C.PRINCIPAL)
    mejores = pd.read_csv(C.RES / "mejores.csv")
    mejores = mejores[mejores.conjunto == C.PRINCIPAL].set_index("modelo")
    pred = pd.read_csv(C.RES / f"pred_{C.PRINCIPAL}.csv", parse_dates=["Mes"])
    conformal = pd.read_csv(C.RES / "conformal.csv")
    cv = cv_interna(datos)
    fijo = datos[datos.Mes == datos.Mes.max()]                       # conjunto fijo de la prueba de ida y vuelta
    explicar = datos[datos.Mes.isin(origenes[-12:])]                 # filas que se explican (último año)
    fondo = datos.sample(30, random_state=C.SEMILLA)
    meses18 = pd.date_range(datos.Mes.max() + pd.DateOffset(months=1), periods=12, freq="MS")
    meta = {"semilla": C.SEMILLA, "python": platform.python_version(),
            "librerias": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "numpy": np.__version__,
                          "pandas": pd.__version__, "joblib": joblib.__version__},
            "entrenamiento": [str(datos.Mes.min().date()), str(datos.Mes.max().date())], "filas": len(datos),
            "objetivo": "indice = ventas del mes / promedio de los 12 meses anteriores; ventas = indice * prom_12m",
            "esquema": {c: ({"tipo": "categórica", "valores": sorted(datos[c].unique())} if datos[c].dtype == object else
                            {"tipo": "numérica", "min": float(datos[c].min()), "max": float(datos[c].max())})
                        for c in FEATURES + ["prom_12m"]},
            "modelos": {}}
    proy, shap = [], []

    for nombre in C.BASE + [C.ORIGINAL]:
        fila = mejores.loc[nombre]
        metodo = "bayes" if nombre == C.ORIGINAL else fila.optimizador
        if nombre == C.ORIGINAL:
            r = optimizar(metodo, PerfilEstacionalJerarquico(), C.REJILLA_ORIGINAL, C.ESPACIO_ORIGINAL, datos, datos[TARGET], cv,
                          n=C.PRESUPUESTO, semilla=C.SEMILLA)
            modelo = PerfilEstacionalJerarquico(**r["params"]).fit(datos, datos[TARGET])
            entrada = lambda t: t                                    # noqa: E731
            pd.DataFrame({"iteracion": range(len(modelo.historial_)), "J": modelo.historial_}).to_csv(ART / "convergencia.csv", index=False)
            pd.DataFrame({"serie": list(modelo.pesos_), "peso": list(modelo.pesos_.values())}).to_csv(ART / "pesos_original.csv", index=False)
        else:
            est, rejilla, espacio = C.MODELOS[nombre]
            pipe = con_pipeline(clone(est))
            r = optimizar(metodo, pipe, prefijo(rejilla), prefijo(espacio), datos[FEATURES], datos[TARGET], cv,
                          n=C.PRESUPUESTO, semilla=C.SEMILLA)
            modelo = clone(pipe).set_params(**r["params"]).fit(datos[FEATURES], datos[TARGET])
            entrada = lambda t: t[FEATURES]                          # noqa: E731
        ruta = ART / f"{slug(nombre)}.joblib"
        joblib.dump(modelo, ruta, compress=3)
        cargado = joblib.load(ruta)
        ida_y_vuelta = bool(np.array_equal(modelo.predict(entrada(fijo)), cargado.predict(entrada(fijo))))

        # radio conforme (cuantil del error absoluto del índice en los 24 meses del bucle externo)
        col = f"{nombre}|{metodo}"
        err = ((pred.Sales - pred[col]).abs() / pred.prom_12m).to_numpy()
        cob = conformal[(conformal.conjunto == C.PRINCIPAL) & (conformal.columna == col)]
        meta["modelos"][nombre] = {
            "archivo": ruta.name, "tamano_kB": round(ruta.stat().st_size / 1024, 1), "optimizador": metodo,
            "hiperparametros": {k.replace("modelo__", ""): v for k, v in r["params"].items()},
            "evaluaciones": len(r["historial"]),
            "bucle_externo": {k: round(float(fila[k]), 4) for k in ["MAE", "MAE_sd", "RMSE", "MASE", "WAPE", "desvio_total_pct", "inferencia_ms"]},
            "radio_conforme_indice": {str(n): float(np.quantile(err, min(1, n * (1 + 1 / err.size)))) for n in (0.8, 0.9)},
            "cobertura_verificada": {str(x.nivel): round(x.cobertura, 3) for x in cob.itertuples()},
            "ida_y_vuelta": ida_y_vuelta}

        p = proyectar(lambda f, m=modelo, e=entrada: a_ventas(m.predict(e(f)), f), panel, meses18)
        proy.append(p.assign(modelo=nombre))
        phis = np.array([shapley_exacto(lambda t, m=modelo, e=entrada: m.predict(e(t)), explicar.iloc[[i]], fondo, FEATURES)[0]
                         for i in range(len(explicar))])
        shap += [{"modelo": nombre, "variable": v, "importancia": float(np.abs(phis[:, j]).mean()),
                  "efecto_medio": float(phis[:, j].mean())} for j, v in enumerate(FEATURES)]
        print(f"{nombre:30s} {metodo:9s} {meta['modelos'][nombre]['tamano_kB']:8.1f} kB · ida y vuelta: {ida_y_vuelta} · "
              f"2018: {p.Proyeccion.sum():,.0f}", flush=True)

    meta["ventas_ultimo_anio"] = float(panel[panel.Mes.dt.year == panel.Mes.dt.year.max()].Sales.sum())
    pd.concat(proy).round(2).to_csv(ART / "proyeccion_2018.csv", index=False)
    pd.DataFrame(shap).to_csv(ART / "shapley_global.csv", index=False)
    (ART / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
