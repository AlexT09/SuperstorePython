"""
model/train_descuentos.py — Modelo del margen según el descuento, para simular la recomendación
"tope de descuento" en la pestaña Recomendaciones aplicadas.

Cada fila es una línea de pedido. El modelo predice el margen (Profit / Sales) a partir del descuento,
la cantidad, la subcategoría, la región y el segmento. Con él se responde: si una línea se hubiera
vendido con menos descuento, ¿qué margen habría tenido?

Flujo (el mismo criterio que train_model.py):
  1. Separa en el tiempo: entrenamiento 2014–2016, prueba 2017.
  2. Ajusta los modelos del curso con GridSearchCV y TimeSeriesSplit; elige por validación cruzada.
  3. Evalúa en 2017 contra la referencia "margen promedio de la subcategoría con ese mismo descuento".
  4. Reentrena el elegido con 2014–2017 y simula cada tope de descuento (los niveles que existen en
     los datos, de 0 % a 80 %) sobre las líneas de 2017, el año más reciente.

Simulación de una línea con descuento d mayor que el tope t:
  - Precio: la venta pasa a Sales × (1 − t) / (1 − d). Las unidades no cambian: en los datos el
    descuento no se relaciona con la cantidad por línea (correlación 0,01).
  - Margen: el real más el cambio que predice el modelo al pasar de d a t.
  - Pedidos perdidos: no se puede estimar con estos datos; la app lo deja como supuesto.

Salidas en model/artifacts/:
  descuentos_resultados.csv   métricas de cada modelo y de la referencia en 2017
  descuentos_escenarios.csv   por categoría × región × tope: ventas y ganancia base y simuladas
  descuentos_resumen.json     modelo elegido, hiperparámetros y métricas

Ejecutar:  python model/train_descuentos.py
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model.features import SEED  # noqa: E402

DATOS = ROOT / "data" / "superstore_transformado.csv"
ART = ROOT / "model" / "artifacts"
ANIO_PRUEBA = 2017
CATEGORICAS = ["Sub-Category", "Region", "Segment"]
NUMERICAS = ["Discount", "Quantity"]
FEATURES = CATEGORICAS + NUMERICAS
TARGET = "margen"
# Solo descuentos que existen en los datos: en valores intermedios (25 %, 35 %...) el modelo extrapola
TOPES = [0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
REFERENCIA = "Referencia (promedio subcategoría × descuento)"

# Modelos del curso con rejillas cortas: aquí hay ~7.000 filas de entrenamiento
MODELOS = {
    "Ridge": (Ridge(), {"alpha": [0.1, 1, 10, 100]}),
    "KNN": (KNeighborsRegressor(), {"n_neighbors": [5, 15, 30], "weights": ["uniform", "distance"]}),
    "Árbol de decisión": (DecisionTreeRegressor(random_state=SEED),
                          {"max_depth": [4, 6, 10], "min_samples_leaf": [5, 20, 50]}),
    "Random Forest": (RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=1),
                      {"max_depth": [6, 10, None], "min_samples_leaf": [5, 20]}),
    "XGBoost": (XGBRegressor(n_estimators=300, random_state=SEED, n_jobs=1, tree_method="hist"),
                {"max_depth": [3, 5], "learning_rate": [0.05, 0.1], "min_child_weight": [1, 5]}),
}


def preprocesador() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
        ("num", StandardScaler(), NUMERICAS),
    ])


def cargar() -> pd.DataFrame:
    d = pd.read_csv(DATOS, parse_dates=["Order_Date"]).sort_values("Order_Date").reset_index(drop=True)
    d[TARGET] = d.Profit / d.Sales
    d["anio"] = d.Order_Date.dt.year
    return d


def metricas(real_m, pred_m, sales) -> dict:
    """Error del margen por línea (puntos %) y de la ganancia total del año."""
    real_m, pred_m, sales = map(np.asarray, (real_m, pred_m, sales))
    ganancia_real, ganancia_pred = (real_m * sales).sum(), (pred_m * sales).sum()
    return {"MAE_margen_pts": float(np.mean(np.abs(real_m - pred_m)) * 100),
            "ganancia_real": float(ganancia_real), "ganancia_pred": float(ganancia_pred),
            "desvio_ganancia_pct": float((ganancia_pred / ganancia_real - 1) * 100)}


def referencia(train, test) -> np.ndarray:
    """Margen promedio de la subcategoría con ese descuento en entrenamiento (si no hay, el de la subcategoría)."""
    por_desc = train.groupby(["Sub-Category", "Discount"])[TARGET].mean()
    por_sub = train.groupby("Sub-Category")[TARGET].mean()
    claves = list(zip(test["Sub-Category"], test.Discount))
    return np.array([por_desc.get(k, por_sub[k[0]]) for k in claves])


def simular(modelo, base: pd.DataFrame) -> pd.DataFrame:
    """Ventas y ganancia de cada categoría × región con cada tope de descuento."""
    pred_actual = modelo.predict(base[FEATURES])
    filas = []
    for t in TOPES:
        cambia = base.Discount > t + 1e-9
        nuevo = base.loc[cambia, FEATURES].assign(Discount=t)
        delta = np.zeros(len(base))
        if cambia.any():
            delta[cambia.to_numpy()] = modelo.predict(nuevo) - pred_actual[cambia.to_numpy()]
        s_nueva = np.where(cambia, base.Sales * (1 - t) / (1 - base.Discount), base.Sales)
        p_nueva = s_nueva * (base[TARGET] + delta)
        sim = base.assign(cambia=cambia, s_nueva=s_nueva, p_nueva=p_nueva)
        for (cat, reg), g in sim.groupby(["Category", "Region"]):
            c = g[g.cambia]
            filas.append({"Category": cat, "Region": reg, "tope": t,
                          "ventas_base": g.Sales.sum(), "ganancia_base": g.Profit.sum(),
                          # Líneas que el tope modifica: antes y después (sin contar pedidos perdidos)
                          "ventas_cambian_antes": c.Sales.sum(), "ganancia_cambian_antes": c.Profit.sum(),
                          "ventas_cambian_despues": c.s_nueva.sum(), "ganancia_cambian_despues": c.p_nueva.sum(),
                          "lineas_cambian": len(c)})
    return pd.DataFrame(filas)


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    ART.mkdir(parents=True, exist_ok=True)
    d = cargar()
    train, test = d[d.anio < ANIO_PRUEBA], d[d.anio == ANIO_PRUEBA]
    cv = TimeSeriesSplit(n_splits=4)
    print(f"Entrenamiento: {len(train)} líneas (2014–2016) · Prueba: {len(test)} líneas (2017)")

    filas = [{"modelo": REFERENCIA, **metricas(test[TARGET], referencia(train, test), test.Sales)}]
    buscadores = {}
    for nombre, (est, grid) in MODELOS.items():
        pipe = Pipeline([("pre", preprocesador()), ("modelo", est)])
        s = GridSearchCV(pipe, {f"modelo__{k}": v for k, v in grid.items()}, cv=cv,
                         scoring="neg_mean_absolute_error", n_jobs=-1)
        t0 = time.perf_counter()
        s.fit(train[FEATURES], train[TARGET])
        m = metricas(test[TARGET], s.predict(test[FEATURES]), test.Sales)
        filas.append({"modelo": nombre, "cv_MAE_margen_pts": -s.best_score_ * 100, **m,
                      "hiperparametros": json.dumps({k.split("__", 1)[1]: v for k, v in s.best_params_.items()},
                                                    ensure_ascii=False)})
        buscadores[nombre] = s
        print(f"  {nombre:18s} CV {-s.best_score_ * 100:5.2f} pts · prueba {m['MAE_margen_pts']:5.2f} pts · "
              f"ganancia {m['desvio_ganancia_pct']:+5.1f} % · {time.perf_counter() - t0:4.1f}s")
    res = pd.DataFrame(filas)
    res.round(4).to_csv(ART / "descuentos_resultados.csv", index=False)

    # Elegido por validación cruzada; se reentrena con toda la historia y se simula sobre 2017
    ganador = min(buscadores, key=lambda n: -buscadores[n].best_score_)
    final = buscadores[ganador].best_estimator_
    final.fit(d[FEATURES], d[TARGET])
    esc = simular(final, test)
    esc.round(2).to_csv(ART / "descuentos_escenarios.csv", index=False)

    fila, ref = res.set_index("modelo").loc[ganador], res.set_index("modelo").loc[REFERENCIA]
    resumen = {"mejor_modelo": ganador, "hiperparametros": json.loads(fila.hiperparametros),
               "cv_MAE_margen_pts": round(float(fila.cv_MAE_margen_pts), 3),
               "prueba": {k: round(float(fila[k]), 3) for k in ["MAE_margen_pts", "desvio_ganancia_pct"]},
               "referencia": {k: round(float(ref[k]), 3) for k in ["MAE_margen_pts", "desvio_ganancia_pct"]},
               "lineas_entrenamiento": len(train), "lineas_prueba": len(test), "anio_base": ANIO_PRUEBA}
    (ART / "descuentos_resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Mejor modelo: {ganador} · artefactos en {ART}")


if __name__ == "__main__":
    main()
