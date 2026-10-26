"""
model/evaluate.py — Métricas de evaluación (en dólares) e importancia de variables.
"""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model.features import FEATURES, NOMBRES, SEED, TARGET


def metricas(real, pred) -> dict:
    """MAE, RMSE y R² por categoría × región × mes, más el MAE como % de la venta media."""
    real, pred = np.asarray(real), np.asarray(pred)
    mae = mean_absolute_error(real, pred)
    return {"MAE": mae, "RMSE": float(np.sqrt(mean_squared_error(real, pred))),
            "R2": r2_score(real, pred), "MAE_pct": mae / real.mean() * 100}


def metricas_total(filas: pd.DataFrame, pred) -> dict:
    """Las mismas métricas sobre el total mensual de la empresa, más el desvío del total del periodo."""
    t = filas.assign(pred=pred).groupby("Mes")[["Sales", "pred"]].sum()
    m = metricas(t.Sales, t.pred)
    m["desvio_total_pct"] = (t.pred.sum() / t.Sales.sum() - 1) * 100
    return {f"total_{k}": v for k, v in m.items()}


def importancia(pipeline, filas: pd.DataFrame) -> pd.DataFrame:
    """Importancia por permutación: cuánto empeora el MAE al desordenar cada variable.

    Funciona igual para los 7 modelos (no depende de coeficientes ni de árboles).
    """
    r = permutation_importance(pipeline, filas[FEATURES], filas[TARGET], n_repeats=30,
                               random_state=SEED, scoring="neg_mean_absolute_error")
    return (pd.DataFrame({"variable": FEATURES, "importancia": r.importances_mean, "std": r.importances_std})
              .assign(nombre=lambda d: d.variable.map(NOMBRES))
              .sort_values("importancia", ascending=False))
