"""
model/features.py — Variables del modelo y preprocesamiento, compartidos por el entrenamiento
(train_model.py) y la proyección (la misma función arma las variables de 2017 y de 2018).

Cada fila es una categoría × región × mes. El modelo no predice las ventas en dólares sino un
índice: cuántas veces el promedio de los últimos 12 meses venderá ese mes (1.3 = 30 % más que el
promedio). Las ventas se recuperan multiplicando el índice por ese promedio. Así una misma regla
sirve para series grandes y pequeñas, y la proyección sigue el nivel reciente de cada serie.

Todas las variables usan solo meses anteriores (shift), por lo que se conocen antes de empezar
el mes que se proyecta: no hay fuga de información del futuro.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
CATEGORICAS = ["Category", "Region"]
NUMERICAS = ["r_lag_12", "r_lag_1", "r_prom_3m", "mes_sin", "mes_cos"]
FEATURES = CATEGORICAS + NUMERICAS
TARGET = "indice"  # Sales / prom_12m

# Nombre legible de cada variable (tablas e importancia de variables)
NOMBRES = {
    "Category": "Categoría",
    "Region": "Región",
    "r_lag_12": "Ventas del mismo mes del año anterior (relativas)",
    "r_lag_1": "Ventas del mes anterior (relativas)",
    "r_prom_3m": "Promedio de los últimos 3 meses (relativo)",
    "mes_sin": "Mes del año (seno)",
    "mes_cos": "Mes del año (coseno)",
}


def agregar_variables(panel: pd.DataFrame) -> pd.DataFrame:
    """Agrega las variables del modelo a la tabla mensual (Category, Region, Mes, Sales).

    Las filas sin 12 meses de historia (el primer año) quedan con NaN y se descartan al entrenar.
    Los meses futuros pueden venir con Sales vacío: sus variables solo dependen de meses previos.
    """
    p = panel.sort_values(["Category", "Region", "Mes"]).copy()
    g = p.groupby(["Category", "Region"]).Sales
    p["lag_12"] = g.shift(12)
    p["lag_1"] = g.shift(1)
    p["prom_3m"] = g.transform(lambda s: s.shift(1).rolling(3).mean())
    p["prom_12m"] = g.transform(lambda s: s.shift(1).rolling(12).mean())
    base = p.prom_12m.where(p.prom_12m > 0)
    for col in ["lag_12", "lag_1", "prom_3m"]:
        p[f"r_{col}"] = p[col] / base
    # Mes como punto en un círculo: diciembre (12) queda junto a enero (1)
    mes = p.Mes.dt.month
    p["mes_sin"] = np.sin(2 * np.pi * mes / 12)
    p["mes_cos"] = np.cos(2 * np.pi * mes / 12)
    p[TARGET] = p.Sales / base
    # Orden cronológico: TimeSeriesSplit asume que las filas siguientes son posteriores
    return p.sort_values(["Mes", "Category", "Region"]).reset_index(drop=True)


def preprocesador() -> ColumnTransformer:
    """One-hot para categoría y región, estandarización para las numéricas (dentro del Pipeline)."""
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
        ("num", StandardScaler(), NUMERICAS),
    ])


def a_ventas(indice, filas: pd.DataFrame) -> np.ndarray:
    """Convierte el índice predicho en ventas (USD); nunca negativas."""
    return np.clip(np.asarray(indice) * filas.prom_12m.to_numpy(), 0, None)
