"""
experiments/robustez.py — Estudios de robustez del protocolo (sección 9), en Sample Superstore.

Compara el modelo original con las dos mejores referencias entrenables, usando en cada origen los
hiperparámetros que ya eligió la búsqueda (tabla_maestra.csv); aquí no se vuelve a optimizar.

Salidas en experiments/resultados/:
    robustez_semillas.csv   MAE de Random Forest y XGBoost con 10 semillas (el original es determinista)
    robustez_ruido.csv      MAE al contaminar la variable objetivo de entrenamiento (ruido y atípicos)
    robustez_muestra.csv    MAE con menos meses de entrenamiento y con menos series
    escalabilidad.csv       tiempo de ajuste al replicar los datos (exponente empírico)

Ejecutar desde la raíz (después de ejecutar.py y analizar.py):  python experiments/robustez.py
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone

import config as C
from ejecutar import cargar, con_pipeline
from model.features import FEATURES, TARGET, a_ventas
from src.original_model import PerfilEstacionalJerarquico

RUIDO = [0.0, 0.1, 0.2, 0.4]
REPETICIONES = 5
VENTANAS = [12, 18, 24]
N_SERIES = [3, 6, 12]


def estimador(modelo, hp):
    """Reconstruye el estimador con los hiperparámetros guardados en la tabla maestra."""
    hp = {k: (None if v == "None" else v) for k, v in json.loads(hp).items()}
    if modelo == C.ORIGINAL:
        return PerfilEstacionalJerarquico(**hp), False
    est = clone(C.EST_HGB if modelo == C.HGB else C.MODELOS[modelo][0]).set_params(**hp)
    return con_pipeline(est), True


def predecir(modelo, hp, tr, te, y=None, semilla=None):
    est, tubo = estimador(modelo, hp)
    if semilla is not None:
        est.set_params(modelo__random_state=semilla)
    y = tr[TARGET] if y is None else y
    return a_ventas(est.fit(tr[FEATURES] if tubo else tr, y).predict(te[FEATURES] if tubo else te), te)


def un_origen(datos, origen, elegidos):
    """Devuelve filas (estudio, modelo, nivel, repetición, suma de errores absolutos, n)."""
    warnings.filterwarnings("ignore")
    tr, te = datos[datos.Mes < origen], datos[datos.Mes == origen]
    real = te.Sales.to_numpy()
    out = []

    def fila(estudio, modelo, nivel, rep, pred, mascara=None):
        e = np.abs(real - pred) if mascara is None else np.abs(real - pred)[mascara]
        out.append((estudio, modelo, nivel, rep, float(e.sum()), int(e.size)))

    for modelo, hp in elegidos.items():
        # ruido multiplicativo y atípicos en la variable objetivo de entrenamiento
        for sd in RUIDO:
            for rep in range(REPETICIONES if sd > 0 else 1):
                rng = np.random.default_rng([C.SEMILLA, rep, int(sd * 100)])
                y = np.clip(tr[TARGET].to_numpy() * (1 + rng.normal(0, sd, len(tr))), 0, None)
                fila("ruido", modelo, f"sd = {sd}", rep, predecir(modelo, hp, tr, te, y))
        for rep in range(REPETICIONES):
            rng = np.random.default_rng([C.SEMILLA, rep, 999])
            y = tr[TARGET].to_numpy().copy()
            y[rng.random(len(tr)) < 0.05] *= 3
            fila("ruido", modelo, "5 % de atípicos (x3)", rep, predecir(modelo, hp, tr, te, y))
        # menos meses de entrenamiento
        for k in VENTANAS:
            trk = tr[tr.Mes >= pd.Timestamp(origen) - pd.DateOffset(months=k)]
            fila("meses", modelo, k, 0, predecir(modelo, hp, trk, te))
        # menos series (las mismas para todos los modelos; se evalúa solo en ellas)
        series = tr[["Category", "Region"]].drop_duplicates().sample(frac=1, random_state=C.SEMILLA)
        for n in N_SERIES:
            sub = series.head(n)
            trn, ten = tr.merge(sub), te.merge(sub)
            p = predecir(modelo, hp, trn, ten)
            out.append(("series", modelo, n, 0, float(np.abs(ten.Sales.to_numpy() - p).sum()), len(ten)))
    return out


def semillas(datos, origen, hps):
    warnings.filterwarnings("ignore")
    tr, te = datos[datos.Mes < origen], datos[datos.Mes == origen]
    return [(m, s, float(np.abs(te.Sales.to_numpy() - predecir(m, hp, tr, te, semilla=s)).sum()), len(te))
            for m, hp in hps.items() for s in C.SEMILLAS]


def escalabilidad(datos):
    """Tiempo de ajuste al replicar los datos 1, 4, 16 y 64 veces (series nuevas con otro nombre)."""
    filas = []
    for k in [1, 4, 16, 64]:
        d = pd.concat([datos.assign(Category=datos.Category + f"_{i}") for i in range(k)], ignore_index=True)
        d = d.sort_values(["Mes", "Category", "Region"])
        for nombre, est, tubo in [(C.ORIGINAL, PerfilEstacionalJerarquico(), False),
                                  ("Ridge", con_pipeline(clone(C.MODELOS["Ridge"][0])), True),
                                  ("Random Forest", con_pipeline(clone(C.MODELOS["Random Forest"][0])), True)]:
            X = d[FEATURES] if tubo else d
            t = []
            for _ in range(3):
                t0 = time.perf_counter(); est.fit(X, d[TARGET]); t.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); est.predict(X); ti = time.perf_counter() - t0
            filas.append({"modelo": nombre, "filas": len(d), "ajuste_ms": np.median(t) * 1e3, "inferencia_ms": ti * 1e3})
    e = pd.DataFrame(filas)
    e["exponente_ajuste"] = e.groupby("modelo").ajuste_ms.transform(
        lambda s: np.polyfit(np.log(e.loc[s.index, "filas"]), np.log(s), 1)[0])
    return e


def main():
    warnings.filterwarnings("ignore")
    panel, datos, origenes = cargar(C.PRINCIPAL)
    maestra = pd.read_csv(C.RES / "tabla_maestra.csv", parse_dates=["origen"])
    maestra = maestra[maestra.conjunto == C.PRINCIPAL]
    mejores = pd.read_csv(C.RES / "mejores.csv")
    mejores = mejores[(mejores.conjunto == C.PRINCIPAL) & mejores.modelo.isin(C.BASE + [C.HGB])].nsmallest(2, "MAE")
    combos = [(C.ORIGINAL, "bayes")] + list(zip(mejores.modelo, mejores.optimizador))

    def hp(modelo, opt, origen):
        return maestra[(maestra.modelo == modelo) & (maestra.optimizador == opt) & (maestra.origen == origen)].hiperparametros.iloc[0]

    res = Parallel(n_jobs=-1)(delayed(un_origen)(datos, o, {m: hp(m, op, o) for m, op in combos}) for o in origenes)
    r = pd.DataFrame([x for y in res for x in y], columns=["estudio", "modelo", "nivel", "rep", "suma", "n"])
    r = r.groupby(["estudio", "modelo", "nivel", "rep"], sort=False).agg(suma=("suma", "sum"), n=("n", "sum")).reset_index()
    r["MAE"] = r.suma / r.n
    r = r.groupby(["estudio", "modelo", "nivel"], sort=False).MAE.agg(["mean", "std"]).reset_index() \
        .rename(columns={"mean": "MAE", "std": "MAE_sd_repeticiones"})
    r[r.estudio == "ruido"].drop(columns="estudio").to_csv(C.RES / "robustez_ruido.csv", index=False)
    r[r.estudio != "ruido"].to_csv(C.RES / "robustez_muestra.csv", index=False)

    azar = ["Random Forest", "XGBoost"]
    res = Parallel(n_jobs=-1)(delayed(semillas)(datos, o, {m: hp(m, "bayes", o) for m in azar}) for o in origenes)
    s = pd.DataFrame([x for y in res for x in y], columns=["modelo", "semilla", "suma", "n"])
    s = s.groupby(["modelo", "semilla"]).sum().reset_index()
    s["MAE"] = s.suma / s.n
    s[["modelo", "semilla", "MAE"]].to_csv(C.RES / "robustez_semillas.csv", index=False)

    escalabilidad(datos).to_csv(C.RES / "escalabilidad.csv", index=False)
    print(r.round(1).to_string(index=False))
    print(s.groupby("modelo").MAE.agg(["mean", "std", "min", "max"]).round(1))


if __name__ == "__main__":
    main()
