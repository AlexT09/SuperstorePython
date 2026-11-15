"""
experiments/config.py — Configuración única de los experimentos (modelos, espacios, presupuesto, semillas).

Todo lo que fija el protocolo pre-registrado (experiments/protocolo.md) está aquí como constantes.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.original_model import PerfilEstacionalJerarquico  # noqa: E402

SEMILLA = 42
RES = ROOT / "experiments" / "resultados"
N_ORIGENES = 24               # meses de evaluación (bucle externo, origen móvil)
PRESUPUESTO = 16              # evaluaciones de hiperparámetros para random, bayes y genético
ALFA, ALFA_MCS = 0.05, 0.10
N_BOOT, BLOQUE = 10000, 3     # bootstrap estacionario
SEMILLAS = list(range(10))    # estabilidad de los modelos con azar

CONJUNTOS = {
    "superstore": ROOT / "data" / "ventas_mensuales.csv",
    "turismo": ROOT / "data" / "adicionales" / "turismo.csv",
    "hospital": ROOT / "data" / "adicionales" / "hospital.csv",
    "m3": ROOT / "data" / "adicionales" / "m3.csv",
}
PRINCIPAL = "superstore"
OPTIMIZADORES = {"superstore": ["grid", "random", "bayes", "genetico"],      # las 28 combinaciones
                 "turismo": ["bayes"], "hospital": ["bayes"], "m3": ["bayes"]}

ORIGINAL = "Perfil jerárquico (original)"
INGENUO = "Ingenuo estacional"
THETA = "Theta"
HGB = "HistGradientBoosting"

# nombre -> (estimador, rejilla de Grid Search, espacio de las búsquedas random / bayes / genético)
MODELOS = {
    "KNN": (KNeighborsRegressor(),
            {"n_neighbors": [3, 5, 10, 15, 20, 30], "weights": ["uniform", "distance"]},
            {"n_neighbors": ("int", 3, 30), "weights": ("cat", ["uniform", "distance"])}),
    "Ridge": (Ridge(),
              {"alpha": [0.01, 0.1, 1, 10, 100]},
              {"alpha": ("log", 0.01, 100)}),
    "Lasso": (Lasso(max_iter=20000),
              {"alpha": [1e-4, 1e-3, 1e-2, 1e-1]},
              {"alpha": ("log", 1e-4, 1e-1)}),
    "Árbol de decisión": (DecisionTreeRegressor(random_state=SEMILLA),
                          {"max_depth": [2, 3, 4, 6], "min_samples_leaf": [1, 5, 10, 20]},
                          {"max_depth": ("int", 2, 6), "min_samples_leaf": ("int", 1, 20)}),
    "Random Forest": (RandomForestRegressor(n_estimators=300, random_state=SEMILLA, n_jobs=1),
                      {"max_depth": [3, 5, None], "min_samples_leaf": [1, 3, 5, 10], "max_features": [1.0, 0.5]},
                      {"max_depth": ("int", 2, 8), "min_samples_leaf": ("int", 1, 10), "max_features": ("real", 0.4, 1.0)}),
    "XGBoost": (XGBRegressor(n_estimators=300, random_state=SEMILLA, n_jobs=1, tree_method="hist"),
                {"max_depth": [2, 3, 4], "learning_rate": [0.03, 0.1], "subsample": [0.8, 1.0], "min_child_weight": [1, 5]},
                {"max_depth": ("int", 2, 4), "learning_rate": ("log", 0.02, 0.2), "subsample": ("real", 0.7, 1.0),
                 "min_child_weight": ("int", 1, 5)}),
    "SVR": (SVR(),
            {"C": [0.1, 1, 10], "epsilon": [0.05, 0.1, 0.2], "gamma": ["scale", 0.05]},
            {"C": ("log", 0.1, 10), "epsilon": ("real", 0.05, 0.2), "gamma": ("log", 0.01, 0.3)}),
}
BASE = list(MODELOS)          # los 7 modelos base del curso

# Método adicional reciente (boosting por histogramas, la familia de LightGBM): solo búsqueda bayesiana
ESPACIO_HGB = {"learning_rate": ("log", 0.02, 0.2), "max_depth": ("int", 2, 4), "min_samples_leaf": ("int", 5, 20)}
EST_HGB = HistGradientBoostingRegressor(max_iter=200, random_state=SEMILLA)

# Modelo original: un solo hiperparámetro propio, lam
REJILLA_ORIGINAL = {"lam": [float(v) for v in np.logspace(-2, 3, 12)]}
ESPACIO_ORIGINAL = {"lam": ("log", 0.01, 1000)}

# Ablación: nombre -> (estimador, ¿se ajusta lam?)
ABLACION = {
    "Peso único (lam infinito)": (PerfilEstacionalJerarquico(pesos="global"), False),
    "Pesos sin agrupar (lam = 0)": (PerfilEstacionalJerarquico(pesos="independiente"), False),
    "Promedio simple (w = 0.5)": (PerfilEstacionalJerarquico(pesos="promedio"), False),
    "Solo perfil (w = 0)": (PerfilEstacionalJerarquico(pesos="cero"), False),
    "Perfil por serie (sin agrupar)": (PerfilEstacionalJerarquico(perfil="serie"), True),
    "Sin perfil (constante)": (PerfilEstacionalJerarquico(perfil="ninguno"), True),
}
