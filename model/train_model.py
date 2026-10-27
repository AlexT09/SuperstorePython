"""
model/train_model.py — Entrena, compara y guarda el modelo de proyección de ventas mensuales.

Flujo:
  1. Lee data/ventas_mensuales.csv (categoría × región × mes) y arma las variables (features.py).
  2. Separa en el tiempo: entrenamiento 2015–2016, prueba 2017 (2014 solo aporta historia).
  3. Para cada uno de los 7 modelos del curso ajusta hiperparámetros con GridSearchCV, usando
     TimeSeriesSplit (cada pliegue valida con meses posteriores).
  4. Evalúa en 2017 contra la referencia "igual que el mismo mes del año anterior".
  5. Elige el mejor modelo por su error de validación cruzada (no por el de prueba, que se usa
     una sola vez para reportar), lo reentrena con 2015–2017 y proyecta 2018 mes a mes.

Salidas en model/artifacts/:
  model.pkl                 pipeline final (preprocesamiento + modelo)
  resultados_modelos.csv    una fila por modelo, más la referencia
  predicciones.csv          real vs. proyectado (entrenamiento y prueba) por modelo
  importancia.csv           importancia de variables del mejor modelo
  prueba_recursiva_2017.csv 2017 proyectado a 12 meses (como 2018) vs. real
  proyeccion_2018.csv       proyección de 2018 por categoría × región × mes
  resumen.json              mejor modelo, hiperparámetros y métricas principales

Ejecutar (después de data/generate_data.py):  python model/train_model.py
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model.evaluate import importancia, metricas, metricas_total  # noqa: E402
from model.features import FEATURES, SEED, TARGET, a_ventas, agregar_variables, preprocesador  # noqa: E402

PANEL = ROOT / "data" / "ventas_mensuales.csv"
ART = ROOT / "model" / "artifacts"
ANIO_PRUEBA = 2017
N_SERIES = 12  # 3 categorías × 4 regiones: filas por mes
# 4 pliegues; cada uno valida con los 3 meses siguientes a su entrenamiento
CV = TimeSeriesSplit(n_splits=4, test_size=3 * N_SERIES)
REFERENCIA = "Referencia (año anterior)"

# nombre: (estimador, rejilla de hiperparámetros para GridSearchCV)
MODELOS = {
    "Ridge": (Ridge(), {"alpha": [0.01, 0.1, 1, 10, 100]}),
    "Lasso": (Lasso(max_iter=20000), {"alpha": [1e-4, 1e-3, 1e-2, 1e-1]}),
    "KNN": (KNeighborsRegressor(), {"n_neighbors": [3, 5, 10, 15, 20, 30], "weights": ["uniform", "distance"]}),
    "Árbol de decisión": (DecisionTreeRegressor(random_state=SEED),
                          {"max_depth": [2, 3, 4, 6], "min_samples_leaf": [1, 5, 10, 20]}),
    "Random Forest": (RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1),
                      {"max_depth": [3, 5, None], "min_samples_leaf": [1, 3, 5, 10], "max_features": [1.0, 0.5]}),
    "XGBoost": (XGBRegressor(n_estimators=300, random_state=SEED, n_jobs=1, tree_method="hist"),
                {"max_depth": [2, 3, 4], "learning_rate": [0.03, 0.1], "subsample": [0.8, 1.0],
                 "min_child_weight": [1, 5]}),
    "SVR": (SVR(), {"C": [0.1, 1, 10], "epsilon": [0.05, 0.1, 0.2], "gamma": ["scale", 0.05]}),
}


def con_prefijo(params: dict) -> dict:
    """Los hiperparámetros viven en el paso 'modelo' del Pipeline."""
    return {f"modelo__{k}": v for k, v in params.items()}


def sin_prefijo(params: dict) -> dict:
    return {k.split("__", 1)[1]: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}


def buscar(estimador, grid, X, y):
    """Ajusta hiperparámetros con GridSearchCV; devuelve el buscador ajustado y su tiempo."""
    pipe = Pipeline([("pre", preprocesador()), ("modelo", estimador)])
    s = GridSearchCV(pipe, con_prefijo(grid), cv=CV, scoring="neg_mean_absolute_error", n_jobs=-1)
    t0 = time.perf_counter()
    s.fit(X, y)
    return s, time.perf_counter() - t0


def proyectar(pipeline, panel: pd.DataFrame, meses) -> pd.DataFrame:
    """Proyección recursiva: cada mes proyectado pasa a ser historia del siguiente."""
    hist = panel[["Category", "Region", "Mes", "Sales"]].copy()
    series = hist[["Category", "Region"]].drop_duplicates()
    for mes in meses:
        hist = pd.concat([hist, series.assign(Mes=mes, Sales=np.nan)], ignore_index=True)
        filas = agregar_variables(hist)
        filas = filas[filas.Mes == mes]
        pred = a_ventas(pipeline.predict(filas[FEATURES]), filas)
        hist = hist[hist.Mes != mes]
        hist = pd.concat([hist, filas[["Category", "Region", "Mes"]].assign(Sales=pred)], ignore_index=True)
    return hist[hist.Mes.isin(meses)].rename(columns={"Sales": "Proyeccion"}).reset_index(drop=True)


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    ART.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL, parse_dates=["Mes"])
    datos = agregar_variables(panel).dropna(subset=FEATURES + [TARGET])
    train = datos[datos.Mes.dt.year < ANIO_PRUEBA]
    test = datos[datos.Mes.dt.year == ANIO_PRUEBA]
    print(f"Entrenamiento: {len(train)} filas ({train.Mes.min():%Y-%m} a {train.Mes.max():%Y-%m}) · "
          f"Prueba: {len(test)} filas ({test.Mes.min():%Y-%m} a {test.Mes.max():%Y-%m})")

    # Referencia: proyectar lo mismo que el mismo mes del año anterior
    filas, preds = [], {REFERENCIA: (train.lag_12.to_numpy(), test.lag_12.to_numpy())}
    filas.append({"modelo": REFERENCIA, **metricas(test.Sales, test.lag_12), **metricas_total(test, test.lag_12)})

    buscadores = {}  # modelo -> GridSearchCV ajustado
    for nombre, (est, grid) in MODELOS.items():
        s, seg = buscar(est, grid, train[FEATURES], train[TARGET])
        pred = a_ventas(s.predict(test[FEATURES]), test)
        filas.append({"modelo": nombre, "evaluaciones": len(s.cv_results_["params"]),
                      "tiempo_s": seg, "cv_MAE_indice": -s.best_score_,
                      "hiperparametros": json.dumps(sin_prefijo(s.best_params_), ensure_ascii=False),
                      "rejilla": json.dumps(grid, ensure_ascii=False),
                      **metricas(test.Sales, pred), **metricas_total(test, pred)})
        print(f"  {nombre:18s} CV {-s.best_score_:.4f} · MAE prueba {filas[-1]['MAE']:7.0f} · {seg:5.1f}s")
        buscadores[nombre] = s
        preds[nombre] = (a_ventas(s.predict(train[FEATURES]), train), pred)

    res = pd.DataFrame(filas)
    res.round(4).to_csv(ART / "resultados_modelos.csv", index=False)

    # Real vs. proyectado de cada modelo, en entrenamiento y prueba
    base = pd.concat([train.assign(conjunto="Entrenamiento"), test.assign(conjunto="Prueba")])
    out = base[["Category", "Region", "Mes", "conjunto", "Sales"]].copy()
    for nombre, (p_tr, p_te) in preds.items():
        out[nombre] = np.concatenate([p_tr, p_te])
    out.round(2).to_csv(ART / "predicciones.csv", index=False)

    # Mejor modelo: menor error de validación cruzada (la prueba no interviene en la elección)
    ganador = min(buscadores, key=lambda n: -buscadores[n].best_score_)
    s = buscadores[ganador]
    importancia(s.best_estimator_, test).round(5).to_csv(ART / "importancia.csv", index=False)

    # Prueba de la proyección a 12 meses: proyectar todo 2017 de forma recursiva, solo con historia
    # hasta 2016 (igual que se hará con 2018). Es la medida honesta del error de la proyección anual.
    meses_prueba = pd.date_range(f"{ANIO_PRUEBA}-01-01", periods=12, freq="MS")
    rec = proyectar(s.best_estimator_, panel[panel.Mes.dt.year < ANIO_PRUEBA], meses_prueba)
    rec = rec.merge(panel, on=["Category", "Region", "Mes"])
    rec.round(2).to_csv(ART / "prueba_recursiva_2017.csv", index=False)
    rec_m = {**metricas(rec.Sales, rec.Proyeccion), **metricas_total(rec, rec.Proyeccion)}

    # Modelo final: mismos hiperparámetros, reentrenado con toda la historia (2015–2017)
    final = Pipeline([("pre", preprocesador()), ("modelo", MODELOS[ganador][0])])
    final.set_params(**s.best_params_)
    final.fit(datos[FEATURES], datos[TARGET])
    joblib.dump(final, ART / "model.pkl")

    meses = pd.date_range(f"{ANIO_PRUEBA + 1}-01-01", periods=12, freq="MS")
    proy = proyectar(final, panel, meses)
    proy.round(2).to_csv(ART / "proyeccion_2018.csv", index=False)

    fila = res[res.modelo == ganador].iloc[0]
    ref = res[res.modelo == REFERENCIA].iloc[0]
    claves = ["MAE", "RMSE", "R2", "MAE_pct", "total_MAE", "total_MAE_pct", "total_R2", "total_desvio_total_pct"]
    resumen = {
        "mejor_modelo": ganador,
        "hiperparametros": json.loads(fila.hiperparametros),
        "prueba": {k: round(float(fila[k]), 4) for k in claves},
        "referencia": {k: round(float(ref[k]), 4) for k in claves},
        "prueba_recursiva": {k: round(float(v), 4) for k, v in rec_m.items()},
        "entrenamiento": f"{train.Mes.min():%Y-%m} a {train.Mes.max():%Y-%m}",
        "prueba_periodo": f"{test.Mes.min():%Y-%m} a {test.Mes.max():%Y-%m}",
        "filas": {"entrenamiento": len(train), "prueba": len(test)},
        "validacion": "TimeSeriesSplit(n_splits=4, test_size=3 meses)",
        "ventas_2017": round(float(panel[panel.Mes.dt.year == 2017].Sales.sum()), 2),
        "proyeccion_2018": round(float(proy.Proyeccion.sum()), 2),
        "semilla": SEED,
    }
    (ART / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMejor modelo: {ganador} · MAE prueba {fila.MAE:,.0f} vs referencia {ref.MAE:,.0f}")
    print(f"Proyección 2018: ${resumen['proyeccion_2018']:,.0f} (2017 real: ${resumen['ventas_2017']:,.0f})")


if __name__ == "__main__":
    main()
